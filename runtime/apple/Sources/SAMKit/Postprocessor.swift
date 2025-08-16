import Foundation
import CoreML
import Accelerate
import CoreGraphics
#if canImport(UIKit)
import UIKit
#endif

/// Handles mask postprocessing for SAM models
public final class Postprocessor {
    
    private let modelSize: Int
    
    public init(modelSize: Int = 1024) {
        self.modelSize = modelSize
    }
    
    /// Process model output into final masks
    public func process(
        maskLogits: MLMultiArray,
        iouPredictions: MLMultiArray,
        transform: TransformParams,
        options: SamOptions
    ) throws -> SamResult {
        
        // Debug: print mask shape
        print("Mask logits shape: \(maskLogits.shape)")
        print("IOU predictions shape: \(iouPredictions.shape)")
        print("IOU predictions count: \(iouPredictions.count)")
        print("Transform: scale=\(transform.scale), padX=\(transform.padX), padY=\(transform.padY)")
        print("Original size: \(transform.originalWidth)x\(transform.originalHeight)")
        
        // Check if mask is already at model resolution (1024x1024) or low resolution (256x256)
        let maskHeight = maskLogits.shape[2].intValue
        let isLowRes = maskHeight <= 256
        
        let processedMasks: MLMultiArray
        if isLowRes {
            // 1. Upsample masks from low resolution to model size
            let upsampled = try upsampleMasks(maskLogits)
            // 2. Remove padding
            let depadded = try removePadding(upsampled, transform: transform)
            // 3. Resize to original image size
            processedMasks = try resizeToOriginal(depadded, transform: transform)
        } else {
            // Already at model resolution, just remove padding and resize
            // 2. Remove padding
            let depadded = try removePadding(maskLogits, transform: transform)
            // 3. Resize to original image size
            processedMasks = try resizeToOriginal(depadded, transform: transform)
        }
        
        // 4. Extract individual masks with scores
        let masks = try extractMasks(
            processedMasks,
            scores: iouPredictions,
            options: options
        )
        
        let scores = masks.map { $0.score }
        return SamResult(masks: masks, scores: scores)
    }
    
    // MARK: - Private Methods
    
    private func upsampleMasks(_ masks: MLMultiArray) throws -> MLMultiArray {
        // HuggingFace SAM2 outputs shape [batch, num_masks, H, W]
        // Standard SAM outputs shape [num_masks, 1, H, W]
        guard masks.shape.count == 4 else {
            throw SamError.postprocessingFailed("Invalid mask shape: \(masks.shape)")
        }
        
        let batchSize = masks.shape[0].intValue
        let secondDim = masks.shape[1].intValue
        let lowResH = masks.shape[2].intValue
        let lowResW = masks.shape[3].intValue
        
        // Check if this is HuggingFace format with grid layout
        // HuggingFace SAM2 puts 3 masks in a 2x2 grid within each 256x256 image
        let isHuggingFaceGrid = (batchSize == 1) && (secondDim == 3) && (lowResH == 256) && (lowResW == 256)
        if isHuggingFaceGrid {
            print("HuggingFace grid layout detected! Each 256x256 contains masks in 2x2 grid")
            return try upsampleHuggingFaceGridMasks(masks)
        }
        
        // Determine the actual number of masks based on shape
        let (actualNumMasks, channelDim): (Int, Int)
        if secondDim > 1 && batchSize == 1 {
            // HuggingFace format: [1, num_masks, H, W]
            actualNumMasks = secondDim
            channelDim = 1
            print("HuggingFace format detected: \(actualNumMasks) masks")
            print("Shape details: batch=\(batchSize), masks=\(secondDim), H=\(lowResH), W=\(lowResW)")
        } else {
            // Standard format: [num_masks, 1, H, W]
            actualNumMasks = batchSize
            channelDim = secondDim
            print("Standard format detected: \(actualNumMasks) masks")
            print("Shape details: batch=\(batchSize), channel=\(secondDim), H=\(lowResH), W=\(lowResW)")
        }
        
        // Create upsampled array - each mask separately
        let upsampled = try MLMultiArray(
            shape: [actualNumMasks as NSNumber, 1, modelSize as NSNumber, modelSize as NSNumber],
            dataType: .float32
        )
        
        let srcPtr = masks.dataPointer.bindMemory(to: Float32.self, capacity: masks.count)
        let dstPtr = upsampled.dataPointer.bindMemory(to: Float32.self, capacity: upsampled.count)
        
        // Debug: print memory layout
        print("Upsampling masks:")
        print("  Source total elements: \(masks.count)")
        print("  Dest total elements: \(upsampled.count)")
        print("  Elements per mask (src): \(lowResH * lowResW)")
        print("  Elements per mask (dst): \(modelSize * modelSize)")
        
        // Bilinear upsampling for each mask
        if secondDim > 1 && batchSize == 1 {
            // HuggingFace format [1, num_masks, H, W]
            // Each mask is at offset: mask_index * H * W
            for n in 0..<actualNumMasks {
                let srcOffset = n * lowResH * lowResW
                let dstOffset = n * modelSize * modelSize
                
                print("  Upsampling mask \(n): src offset=\(srcOffset), dst offset=\(dstOffset)")
                
                let maskSrc = srcPtr.advanced(by: srcOffset)
                let maskDst = dstPtr.advanced(by: dstOffset)
                
                bilinearUpsample(
                    src: maskSrc,
                    dst: maskDst,
                    srcWidth: lowResW,
                    srcHeight: lowResH,
                    dstWidth: modelSize,
                    dstHeight: modelSize
                )
            }
        } else {
            // Standard format [num_masks, 1, H, W]
            for n in 0..<actualNumMasks {
                let srcOffset = n * channelDim * lowResH * lowResW
                let dstOffset = n * modelSize * modelSize
                
                print("  Upsampling mask \(n): src offset=\(srcOffset), dst offset=\(dstOffset)")
                
                let maskSrc = srcPtr.advanced(by: srcOffset)
                let maskDst = dstPtr.advanced(by: dstOffset)
                
                bilinearUpsample(
                    src: maskSrc,
                    dst: maskDst,
                    srcWidth: lowResW,
                    srcHeight: lowResH,
                    dstWidth: modelSize,
                    dstHeight: modelSize
                )
            }
        }
        
        return upsampled
    }
    
    private func upsampleHuggingFaceGridMasks(_ masks: MLMultiArray) throws -> MLMultiArray {
        // HuggingFace format: [1, 3, 256, 256] where each 256x256 contains a 2x2 grid
        // We need to extract each 128x128 quadrant as a separate mask
        
        let numMasks = masks.shape[1].intValue  // Should be 3
        let gridSize = masks.shape[2].intValue  // Should be 256
        let halfSize = gridSize / 2  // 128
        
        print("Extracting \(numMasks) masks from 2x2 grid layout (each \(halfSize)x\(halfSize))")
        
        // Create upsampled array for 3 separate masks
        let upsampled = try MLMultiArray(
            shape: [numMasks as NSNumber, 1, modelSize as NSNumber, modelSize as NSNumber],
            dataType: .float32
        )
        
        let srcPtr = masks.dataPointer.bindMemory(to: Float32.self, capacity: masks.count)
        let dstPtr = upsampled.dataPointer.bindMemory(to: Float32.self, capacity: upsampled.count)
        
        // For each of the 3 masks
        for maskIdx in 0..<numMasks {
            // Get the source data for this mask (256x256 grid)
            let maskSrcBase = srcPtr.advanced(by: maskIdx * gridSize * gridSize)
            
            // Determine which quadrant to extract
            // Mask 0: top-left (0,0)
            // Mask 1: top-right (0,128) 
            // Mask 2: bottom-left (128,0)
            let (quadrantY, quadrantX) = getQuadrantPosition(maskIndex: maskIdx)
            
            print("  Mask \(maskIdx): extracting quadrant at (\(quadrantX), \(quadrantY))")
            
            // Create temporary buffer for the extracted quadrant
            var quadrantData = [Float32](repeating: 0, count: halfSize * halfSize)
            
            // Extract the quadrant from the grid
            for y in 0..<halfSize {
                for x in 0..<halfSize {
                    let srcY = quadrantY + y
                    let srcX = quadrantX + x
                    let srcIdx = srcY * gridSize + srcX
                    let dstIdx = y * halfSize + x
                    quadrantData[dstIdx] = maskSrcBase[srcIdx]
                }
            }
            
            // Upsample the quadrant to model size
            let maskDst = dstPtr.advanced(by: maskIdx * modelSize * modelSize)
            quadrantData.withUnsafeBufferPointer { buffer in
                bilinearUpsample(
                    src: UnsafeMutablePointer(mutating: buffer.baseAddress!),
                    dst: maskDst,
                    srcWidth: halfSize,
                    srcHeight: halfSize,
                    dstWidth: modelSize,
                    dstHeight: modelSize
                )
            }
        }
        
        return upsampled
    }
    
    private func getQuadrantPosition(maskIndex: Int) -> (y: Int, x: Int) {
        // Map mask index to quadrant position in 2x2 grid
        switch maskIndex {
        case 0: return (0, 0)      // Top-left
        case 1: return (0, 128)    // Top-right
        case 2: return (128, 0)    // Bottom-left
        default: return (128, 128) // Bottom-right (for mask 3 if exists)
        }
    }
    
    private func removePadding(_ masks: MLMultiArray, transform: TransformParams) throws -> MLMultiArray {
        let numMasks = masks.shape[0].intValue
        let channels = masks.shape.count > 1 ? masks.shape[1].intValue : 1
        let maskHeight = masks.shape[2].intValue
        let maskWidth = masks.shape[3].intValue
        
        // Check if mask is at model resolution
        guard maskHeight == modelSize && maskWidth == modelSize else {
            // If not at model size, no padding to remove
            print("No padding removal needed - mask size: \(maskWidth)x\(maskHeight)")
            return masks
        }
        
        // Calculate the actual image size in model space
        let scaledWidth = Int(Float(transform.originalWidth) * transform.scale)
        let scaledHeight = Int(Float(transform.originalHeight) * transform.scale)
        
        // Calculate padding
        let padLeft = Int(transform.padX)
        let padTop = Int(transform.padY)
        
        print("Removing padding: scaledSize=\(scaledWidth)x\(scaledHeight), pad=(left:\(padLeft), top:\(padTop))")
        
        // Create depadded array
        let depadded = try MLMultiArray(
            shape: [numMasks as NSNumber, channels as NSNumber, scaledHeight as NSNumber, scaledWidth as NSNumber],
            dataType: .float32
        )
        
        let srcPtr = masks.dataPointer.bindMemory(to: Float32.self, capacity: masks.count)
        let dstPtr = depadded.dataPointer.bindMemory(to: Float32.self, capacity: depadded.count)
        
        // Copy unpadded region for each mask
        for n in 0..<numMasks {
            for c in 0..<channels {
                let srcOffset = (n * channels + c) * maskHeight * maskWidth
                let dstOffset = (n * channels + c) * scaledHeight * scaledWidth
                
                let maskSrc = srcPtr.advanced(by: srcOffset)
                let maskDst = dstPtr.advanced(by: dstOffset)
                
                for y in 0..<scaledHeight {
                    for x in 0..<scaledWidth {
                        let srcIdx = (y + padTop) * maskWidth + (x + padLeft)
                        let dstIdx = y * scaledWidth + x
                        maskDst[dstIdx] = maskSrc[srcIdx]
                    }
                }
            }
        }
        
        return depadded
    }
    
    private func resizeToOriginal(_ masks: MLMultiArray, transform: TransformParams) throws -> MLMultiArray {
        let numMasks = masks.shape[0].intValue
        let scaledHeight = masks.shape[2].intValue
        let scaledWidth = masks.shape[3].intValue
        
        // Create final array at original resolution
        let finalMasks = try MLMultiArray(
            shape: [numMasks as NSNumber, 1, 
                   transform.originalHeight as NSNumber, transform.originalWidth as NSNumber],
            dataType: .float32
        )
        
        let srcPtr = masks.dataPointer.bindMemory(to: Float32.self, capacity: masks.count)
        let dstPtr = finalMasks.dataPointer.bindMemory(to: Float32.self, capacity: finalMasks.count)
        
        // Resize each mask to original size
        for n in 0..<numMasks {
            let maskSrc = srcPtr.advanced(by: n * scaledHeight * scaledWidth)
            let maskDst = dstPtr.advanced(by: n * transform.originalHeight * transform.originalWidth)
            
            bilinearUpsample(
                src: maskSrc,
                dst: maskDst,
                srcWidth: scaledWidth,
                srcHeight: scaledHeight,
                dstWidth: transform.originalWidth,
                dstHeight: transform.originalHeight
            )
        }
        
        return finalMasks
    }
    
    private func extractMasks(
        _ masks: MLMultiArray,
        scores: MLMultiArray,
        options: SamOptions
    ) throws -> [SamMask] {
        
        let numMasks = masks.shape[0].intValue
        let channels = masks.shape[1].intValue
        let height = masks.shape[2].intValue
        let width = masks.shape[3].intValue
        
        print("Extracting \(numMasks) masks of size \(width)x\(height)")
        print("Scores shape: \(scores.shape), count: \(scores.count)")
        
        // Check if scores is 2D array [1, N] and adjust indexing
        let isScores2D = scores.shape.count == 2 && scores.shape[0].intValue == 1
        
        // Check if scores count matches masks count
        let scoresCount = isScores2D ? scores.shape[1].intValue : scores.count
        if scoresCount < numMasks {
            print("Warning: scores count (\(scoresCount)) is less than masks count (\(numMasks))")
        }
        
        let maskPtr = masks.dataPointer.bindMemory(to: Float32.self, capacity: masks.count)
        let scorePtr = scores.dataPointer.bindMemory(to: Float32.self, capacity: scores.count)
        
        // Get indices sorted by score
        let sortedIndices = sortMasksByScore(scores: scorePtr, count: min(numMasks, scoresCount))
        
        // Print all scores for debugging
        print("All scores:")
        
        for i in 0..<min(numMasks, scoresCount) {
            let scoreValue: Float
            if isScores2D {
                // For [1, N] shape, use 2D indexing
                scoreValue = scores[[0, i as NSNumber] as [NSNumber]].floatValue
            } else {
                // For [N] shape, use 1D indexing
                scoreValue = scorePtr[i]
            }
            print("  Score[\(i)] = \(scoreValue)")
        }
        
        // Determine how many masks to return
        // HuggingFace models may output 4 masks (including background), we typically want 3
        let effectiveNumMasks = min(numMasks, 3)  // Limit to 3 masks maximum
        let masksToReturn = options.multimaskOutput ? min(options.maxMasks, effectiveNumMasks) : 1
        
        print("Num masks in output: \(numMasks), returning: \(masksToReturn)")
        
        var resultMasks: [SamMask] = []
        
        for i in 0..<masksToReturn {
            // Ensure we have a valid index
            guard i < sortedIndices.count else {
                print("Warning: Trying to access index \(i) but only have \(sortedIndices.count) sorted indices")
                break
            }
            
            let idx = sortedIndices[i]
            
            // Ensure idx is valid for scores
            guard idx < scoresCount else {
                print("Warning: Index \(idx) is out of bounds for scores array (count: \(scoresCount))")
                continue
            }
            
            // Calculate offset for this specific mask
            // masks shape is [numMasks, 1, height, width]
            let maskOffset = idx * channels * height * width
            print("Extracting mask \(i) (idx=\(idx)): offset=\(maskOffset), size=\(width)x\(height)")
            
            let maskData = maskPtr.advanced(by: maskOffset)
            
            // Prepare logits if requested
            var logits: [Float]? = nil
            if options.returnLogits {
                logits = Array(UnsafeBufferPointer(start: maskData, count: height * width))
            }
            
            // Convert to alpha mask for this specific mask only
            let alpha = createAlphaMaskForSingle(
                from: maskData,
                width: width,
                height: height,
                threshold: options.maskThreshold
            )
            
            let cgImage = createCGImage(from: alpha, width: width, height: height)
            
            // Get score with proper indexing
            var score: Float
            if isScores2D {
                // For [1, N] shape, use 2D indexing
                score = scores[[0, idx as NSNumber] as [NSNumber]].floatValue
            } else {
                // For [N] shape, use 1D indexing  
                score = scorePtr[idx]
            }
            
            // Validate and sanitize score
            if score.isNaN || score.isInfinite || score < 0.0 {
                print("Warning: Invalid score \(score) for mask \(i), using 0.0")
                score = 0.0
            }
            
            let mask = SamMask(
                width: width,
                height: height,
                logits: logits,
                alpha: alpha,
                score: score,
                cgImage: cgImage
            )
            
            // Debug: save mask as PNG and print raw data sample
            #if DEBUG
            // Print first few values of mask data
            print("Mask \(i) first 10 values:")
            for j in 0..<min(10, height * width) {
                print("  [\(j)] = \(maskData[j])")
            }
            
            if let data = UIImage(cgImage: cgImage).pngData() {
                let path = FileManager.default.temporaryDirectory.appendingPathComponent("mask_\(i)_\(width)x\(height).png")
                try? data.write(to: path)
                print("Saved mask \(i) to: \(path.path)")
            }
            #endif
            
            resultMasks.append(mask)
        }
        
        return resultMasks
    }
    
    private func bilinearUpsample(
        src: UnsafeMutablePointer<Float32>,
        dst: UnsafeMutablePointer<Float32>,
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int
    ) {
        let xScale = Float(srcWidth) / Float(dstWidth)
        let yScale = Float(srcHeight) / Float(dstHeight)
        
        for y in 0..<dstHeight {
            let srcY = Float(y) * yScale
            let y0 = Int(srcY)
            let y1 = min(y0 + 1, srcHeight - 1)
            let dy = srcY - Float(y0)
            
            for x in 0..<dstWidth {
                let srcX = Float(x) * xScale
                let x0 = Int(srcX)
                let x1 = min(x0 + 1, srcWidth - 1)
                let dx = srcX - Float(x0)
                
                // Bilinear interpolation
                let v00 = src[y0 * srcWidth + x0]
                let v01 = src[y0 * srcWidth + x1]
                let v10 = src[y1 * srcWidth + x0]
                let v11 = src[y1 * srcWidth + x1]
                
                let v0 = v00 * (1 - dx) + v01 * dx
                let v1 = v10 * (1 - dx) + v11 * dx
                let v = v0 * (1 - dy) + v1 * dy
                
                dst[y * dstWidth + x] = v
            }
        }
    }
    
    private func sortMasksByScore(scores: UnsafeMutablePointer<Float32>, count: Int) -> [Int] {
        var indices = Array(0..<count)
        indices.sort { scores[$0] > scores[$1] }
        return indices
    }
    
    private func createAlphaMaskForSingle(
        from logits: UnsafeMutablePointer<Float32>,
        width: Int,
        height: Int,
        threshold: Float
    ) -> Data {
        var alphaData = Data(count: width * height)
        
        alphaData.withUnsafeMutableBytes { buffer in
            guard let ptr = buffer.bindMemory(to: UInt8.self).baseAddress else { return }
            
            // Process exactly width * height elements for this single mask
            if threshold > 0 {
                // Apply threshold
                for i in 0..<(width * height) {
                    ptr[i] = logits[i] > threshold ? 255 : 0
                }
            } else {
                // Convert logits to alpha using sigmoid
                for i in 0..<(width * height) {
                    let logitValue = logits[i]
                    // Clamp logit values to prevent overflow/underflow
                    let clampedLogit = min(50, max(-50, logitValue))
                    let sigmoid = 1.0 / (1.0 + exp(-clampedLogit))
                    // Ensure we have a valid value before converting to Int
                    let alphaFloat = sigmoid * 255.0
                    if alphaFloat.isNaN || alphaFloat.isInfinite {
                        ptr[i] = 0
                    } else {
                        ptr[i] = UInt8(min(255, max(0, Int(alphaFloat + 0.5))))
                    }
                }
            }
        }
        
        return alphaData
    }
    
    private func createAlphaMask(
        from logits: UnsafeMutablePointer<Float32>,
        width: Int,
        height: Int,
        threshold: Float
    ) -> Data {
        var alphaData = Data(count: width * height)
        
        alphaData.withUnsafeMutableBytes { buffer in
            guard let ptr = buffer.bindMemory(to: UInt8.self).baseAddress else { return }
            
            if threshold > 0 {
                // Apply threshold
                for i in 0..<(width * height) {
                    ptr[i] = logits[i] > threshold ? 255 : 0
                }
            } else {
                // Convert logits to alpha using sigmoid
                for i in 0..<(width * height) {
                    let logitValue = logits[i]
                    // Clamp logit values to prevent overflow/underflow
                    let clampedLogit = min(50, max(-50, logitValue))
                    let sigmoid = 1.0 / (1.0 + exp(-clampedLogit))
                    // Ensure we have a valid value before converting to Int
                    let alphaFloat = sigmoid * 255.0
                    if alphaFloat.isNaN || alphaFloat.isInfinite {
                        ptr[i] = 0
                    } else {
                        ptr[i] = UInt8(min(255, max(0, Int(alphaFloat + 0.5))))
                    }
                }
            }
        }
        
        return alphaData
    }
    
    private func createCGImage(from alpha: Data, width: Int, height: Int) -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        
        // Create RGBA data
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        
        alpha.withUnsafeBytes { buffer in
            guard let ptr = buffer.bindMemory(to: UInt8.self).baseAddress else { return }
            
            for y in 0..<height {
                for x in 0..<width {
                    let alphaValue = ptr[y * width + x]
                    let pixelIndex = y * bytesPerRow + x * bytesPerPixel
                    
                    // Set blue color with alpha (for better visibility)
                    pixelData[pixelIndex] = 30       // R
                    pixelData[pixelIndex + 1] = 144  // G  
                    pixelData[pixelIndex + 2] = 255  // B (DodgerBlue)
                    pixelData[pixelIndex + 3] = alphaValue // A
                }
            }
        }
        
        // Create CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let provider = CGDataProvider(data: NSData(bytes: &pixelData, length: pixelData.count)),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            // Fallback: create a simple black image
            let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            )!
            context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            return context.makeImage()!
        }
        
        return cgImage
    }
}

// MARK: - Extensions for Mask Visualization

extension SamMask {
    /// Convert mask to CGImage for visualization
    public func toCGImage() -> CGImage? {
        let width = self.width
        let height = self.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        
        // Create RGBA data
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        
        alpha.withUnsafeBytes { buffer in
            guard let ptr = buffer.bindMemory(to: UInt8.self).baseAddress else { return }
            
            for y in 0..<height {
                for x in 0..<width {
                    let alphaValue = ptr[y * width + x]
                    let pixelIndex = y * bytesPerRow + x * bytesPerPixel
                    
                    // Set white color with alpha
                    pixelData[pixelIndex] = 255      // R
                    pixelData[pixelIndex + 1] = 255  // G
                    pixelData[pixelIndex + 2] = 255  // B
                    pixelData[pixelIndex + 3] = alphaValue // A
                }
            }
        }
        
        // Create CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let provider = CGDataProvider(data: NSData(bytes: &pixelData, length: pixelData.count)) else {
            return nil
        }
        
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
    
    /// Apply color overlay to mask
    public func toColoredCGImage(color: CGColor, backgroundColor: CGColor? = nil) -> CGImage? {
        let width = self.width
        let height = self.height
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            return nil
        }
        
        // Fill background if provided
        if let backgroundColor = backgroundColor {
            context.setFillColor(backgroundColor)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        }
        
        // Draw mask with color
        if let maskImage = toCGImage() {
            context.saveGState()
            context.setBlendMode(.normal)
            context.setAlpha(0.5) // Semi-transparent overlay
            context.setFillColor(color)
            context.clip(to: CGRect(x: 0, y: 0, width: width, height: height), mask: maskImage)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.restoreGState()
        }
        
        return context.makeImage()
    }
}