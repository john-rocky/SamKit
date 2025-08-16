import Foundation
import CoreGraphics
import CoreML
import Accelerate

/// Handles image preprocessing for SAM models
public final class Preprocessor {
    
    private let modelSize: Int
    private let mean: [Float]
    private let std: [Float]
    
    public init(modelSize: Int = 1024) {
        self.modelSize = modelSize
        // Default SAM normalization parameters
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    }
    
    /// Process image for model input
    public func process(_ image: CGImage) throws -> (MLMultiArray, TransformParams) {
        // Calculate transform parameters
        let transform = computeTransform(
            originalWidth: image.width,
            originalHeight: image.height
        )
        
        // Resize and pad image
        guard let resizedImage = resizeAndPad(image, transform: transform) else {
            throw SamError.preprocessingFailed("Failed to resize and pad image")
        }
        
        // Convert to MLMultiArray
        let array = try imageToMLMultiArray(resizedImage)
        
        // Normalize
        normalize(array)
        
        return (array, transform)
    }
    
    /// Encode point prompts
    public func encodePoints(_ points: [SamPoint], transform: TransformParams) throws -> (MLMultiArray, MLMultiArray) {
        let count = max(1, points.count)  // At least 1 point for shape consistency
        
        // Create coordinate array (N, 2)
        let coords = try MLMultiArray(shape: [1, count as NSNumber, 2], dataType: .float32)
        let coordsPtr = coords.dataPointer.bindMemory(to: Float32.self, capacity: coords.count)
        
        // Create label array (N,)
        let labels = try MLMultiArray(shape: [1, count as NSNumber], dataType: .float32)
        let labelsPtr = labels.dataPointer.bindMemory(to: Float32.self, capacity: labels.count)
        
        if points.isEmpty {
            // Fill with dummy point at center
            coordsPtr[0] = Float(modelSize / 2)
            coordsPtr[1] = Float(modelSize / 2)
            labelsPtr[0] = -1  // Invalid label
        } else {
            for (i, point) in points.enumerated() {
                // Transform to model coordinates
                let modelPoint = transform.toModel(point)
                coordsPtr[i * 2] = Float(modelPoint.x)
                coordsPtr[i * 2 + 1] = Float(modelPoint.y)
                labelsPtr[i] = Float(point.label.rawValue)
            }
        }
        
        return (coords, labels)
    }
    
    /// Encode mask input
    public func encodeMask(_ mask: SamMaskRef, transform: TransformParams) throws -> MLMultiArray {
        let maskArray = try MLMultiArray(shape: [1, 1, 256, 256], dataType: .float32)
        let ptr = maskArray.dataPointer.bindMemory(to: Float32.self, capacity: maskArray.count)
        
        // Resample mask to 256x256
        // This is simplified - real implementation would use proper resampling
        let scaleX = Float(mask.width) / 256.0
        let scaleY = Float(mask.height) / 256.0
        
        for y in 0..<256 {
            for x in 0..<256 {
                let srcX = Int(Float(x) * scaleX)
                let srcY = Int(Float(y) * scaleY)
                let srcIdx = srcY * mask.width + srcX
                
                let value: Float
                if let logits = mask.logits, srcIdx < logits.count {
                    value = logits[srcIdx]
                } else if srcIdx < mask.alpha.count {
                    value = Float(mask.alpha[srcIdx]) / 255.0
                } else {
                    value = 0
                }
                
                ptr[y * 256 + x] = value
            }
        }
        
        return maskArray
    }
    
    /// Encode box input for SAM decoder
    public func encodeBox(_ box: SamBox, transform: TransformParams) throws -> MLMultiArray {
        let boxArray = try MLMultiArray(shape: [1, 1, 4], dataType: .float32)
        
        // Transform box coordinates to model space
        let x0 = Float(box.x0) * transform.scale + transform.padX
        let y0 = Float(box.y0) * transform.scale + transform.padY
        let x1 = Float(box.x1) * transform.scale + transform.padX
        let y1 = Float(box.y1) * transform.scale + transform.padY
        
        boxArray[[0, 0, 0] as [NSNumber]] = NSNumber(value: x0)
        boxArray[[0, 0, 1] as [NSNumber]] = NSNumber(value: y0)
        boxArray[[0, 0, 2] as [NSNumber]] = NSNumber(value: x1)
        boxArray[[0, 0, 3] as [NSNumber]] = NSNumber(value: y1)
        
        return boxArray
    }
    
    // MARK: - Private Methods
    
    internal func computeTransform(originalWidth: Int, originalHeight: Int) -> TransformParams {
        let longSide = max(originalWidth, originalHeight)
        let scale = Float(modelSize) / Float(longSide)
        
        let scaledWidth = Int(Float(originalWidth) * scale)
        let scaledHeight = Int(Float(originalHeight) * scale)
        
        let padX = Float(modelSize - scaledWidth) / 2.0
        let padY = Float(modelSize - scaledHeight) / 2.0
        
        return TransformParams(
            scale: scale,
            padX: padX,
            padY: padY,
            originalWidth: originalWidth,
            originalHeight: originalHeight,
            modelSize: modelSize
        )
    }
    
    internal func resizeAndPad(_ image: CGImage, transform: TransformParams) -> CGImage? {
        let scaledWidth = Int(Float(image.width) * transform.scale)
        let scaledHeight = Int(Float(image.height) * transform.scale)
        
        // Create context for padded image
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        
        guard let context = CGContext(
            data: nil,
            width: modelSize,
            height: modelSize,
            bitsPerComponent: 8,
            bytesPerRow: modelSize * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            return nil
        }
        
        // Fill with black (padding)
        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: modelSize, height: modelSize))
        
        // Draw scaled image at center
        let drawRect = CGRect(
            x: Int(transform.padX),
            y: Int(transform.padY),
            width: scaledWidth,
            height: scaledHeight
        )
        
        context.draw(image, in: drawRect)
        
        return context.makeImage()
    }
    
    private func imageToMLMultiArray(_ image: CGImage) throws -> MLMultiArray {
        // Create MLMultiArray in CHW format (3, H, W)
        let array = try MLMultiArray(shape: [3, modelSize as NSNumber, modelSize as NSNumber], dataType: .float32)
        
        // Get pixel data
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw SamError.preprocessingFailed("Failed to create bitmap context")
        }
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Copy to MLMultiArray in CHW format
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let channelSize = modelSize * modelSize
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * bytesPerPixel
                
                // R channel
                ptr[0 * channelSize + y * width + x] = Float(pixelData[pixelIndex])
                // G channel
                ptr[1 * channelSize + y * width + x] = Float(pixelData[pixelIndex + 1])
                // B channel
                ptr[2 * channelSize + y * width + x] = Float(pixelData[pixelIndex + 2])
            }
        }
        
        return array
    }
    
    private func normalize(_ array: MLMultiArray) {
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let channelSize = modelSize * modelSize
        
        // Normalize each channel
        for c in 0..<3 {
            let channelStart = c * channelSize
            for i in 0..<channelSize {
                let idx = channelStart + i
                ptr[idx] = (ptr[idx] - mean[c]) / std[c]
            }
        }
    }
}

/// Transform parameters for coordinate mapping
public struct TransformParams {
    public let scale: Float
    public let padX: Float
    public let padY: Float
    public let originalWidth: Int
    public let originalHeight: Int
    public let modelSize: Int
    
    /// Convert point from image coordinates to model coordinates
    public func toModel(_ point: SamPoint) -> SamPoint {
        return SamPoint(
            x: point.x * CGFloat(scale) + CGFloat(padX),
            y: point.y * CGFloat(scale) + CGFloat(padY),
            label: point.label
        )
    }
    
    /// Convert point from model coordinates to image coordinates
    public func toImage(_ point: SamPoint) -> SamPoint {
        return SamPoint(
            x: (point.x - CGFloat(padX)) / CGFloat(scale),
            y: (point.y - CGFloat(padY)) / CGFloat(scale),
            label: point.label
        )
    }
}