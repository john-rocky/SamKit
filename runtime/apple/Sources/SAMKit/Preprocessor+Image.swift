import Foundation
import CoreGraphics
import CoreML
import Vision

extension Preprocessor {
    
    /// Process image for SAM2 models that expect CGImage input
    public func processForSAM2(_ image: CGImage) throws -> (CGImage, TransformParams) {
        // Calculate transform parameters
        let transform = computeTransform(
            originalWidth: image.width,
            originalHeight: image.height
        )
        
        // Resize and pad image
        guard let processedImage = resizeAndPad(image, transform: transform) else {
            throw SamError.preprocessingFailed("Failed to resize and pad image")
        }
        
        return (processedImage, transform)
    }
    
    /// Create an MLFeatureValue from CGImage for SAM2
    public func createImageFeature(_ image: CGImage) throws -> MLFeatureValue {
        // Create CVPixelBuffer from CGImage
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            image.width,
            image.height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw SamError.preprocessingFailed("Failed to create pixel buffer")
        }
        
        // Draw image into pixel buffer
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        )
        
        context?.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        
        return MLFeatureValue(pixelBuffer: buffer)
    }
}