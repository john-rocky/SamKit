import Foundation
import CoreGraphics
import CoreML
import Accelerate

/// Preprocesses images for YOLO-World detection and postprocesses detector output.
public final class GroundingPreprocessor {

    private let inputSize: Int

    public init(inputSize: Int = 640) {
        self.inputSize = inputSize
    }

    // MARK: - Image Preprocessing

    /// Preprocess image for YOLO-World: resize with letterboxing to inputSize x inputSize
    /// Returns the preprocessed MLMultiArray and transform parameters for coordinate mapping
    public func process(_ image: CGImage) throws -> (MLMultiArray, GroundingTransform) {
        let origW = image.width
        let origH = image.height

        // Compute scale (fit long side to inputSize)
        let scale = Float(inputSize) / Float(max(origW, origH))
        let scaledW = Int(Float(origW) * scale)
        let scaledH = Int(Float(origH) * scale)
        let padX = (inputSize - scaledW) / 2
        let padY = (inputSize - scaledH) / 2

        let transform = GroundingTransform(
            scale: scale,
            padX: Float(padX),
            padY: Float(padY),
            originalWidth: origW,
            originalHeight: origH,
            inputSize: inputSize
        )

        // Draw resized image onto padded canvas
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        guard let ctx = CGContext(
            data: nil,
            width: inputSize,
            height: inputSize,
            bitsPerComponent: 8,
            bytesPerRow: inputSize * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw GroundingError.detectionFailed("Failed to create graphics context")
        }

        // Black background (padding)
        ctx.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0))
        ctx.fill(CGRect(x: 0, y: 0, width: inputSize, height: inputSize))

        // Draw image at center (Core Graphics has flipped Y)
        ctx.draw(image, in: CGRect(x: padX, y: padY, width: scaledW, height: scaledH))

        guard let pixelData = ctx.data else {
            throw GroundingError.detectionFailed("Failed to get pixel data")
        }

        let pixels = pixelData.bindMemory(to: UInt8.self, capacity: inputSize * inputSize * 4)

        // Convert to MLMultiArray [1, 3, H, W], normalized to [0, 1]
        let array = try MLMultiArray(
            shape: [1, 3, inputSize as NSNumber, inputSize as NSNumber],
            dataType: .float32
        )
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let planeSize = inputSize * inputSize

        for y in 0..<inputSize {
            for x in 0..<inputSize {
                let srcIdx = (y * inputSize + x) * 4
                let dstIdx = y * inputSize + x
                ptr[dstIdx] = Float32(pixels[srcIdx]) / 255.0                   // R
                ptr[planeSize + dstIdx] = Float32(pixels[srcIdx + 1]) / 255.0   // G
                ptr[2 * planeSize + dstIdx] = Float32(pixels[srcIdx + 2]) / 255.0 // B
            }
        }

        return (array, transform)
    }

    // MARK: - Coordinate Transforms

    /// Convert bounding box from detector output (model space) to original image coordinates
    public func toImageCoordinates(_ box: DetectorBox, transform: GroundingTransform) -> DetectorBox {
        return DetectorBox(
            x0: (box.x0 - transform.padX) / transform.scale,
            y0: (box.y0 - transform.padY) / transform.scale,
            x1: (box.x1 - transform.padX) / transform.scale,
            y1: (box.y1 - transform.padY) / transform.scale
        )
    }

    // MARK: - Non-Maximum Suppression

    /// Apply NMS to filter overlapping detections
    public func nms(
        boxes: [DetectorBox],
        scores: [Float],
        labels: [Int],
        iouThreshold: Float
    ) -> [Int] {
        guard !boxes.isEmpty else { return [] }

        // Sort by score descending
        let indices = scores.enumerated()
            .sorted { $0.element > $1.element }
            .map { $0.offset }

        var kept: [Int] = []
        var suppressed = Set<Int>()

        for i in indices {
            if suppressed.contains(i) { continue }
            kept.append(i)

            for j in indices {
                if suppressed.contains(j) || j == i { continue }
                // Only suppress same-class detections
                if labels[i] != labels[j] { continue }
                if iou(boxes[i], boxes[j]) > iouThreshold {
                    suppressed.insert(j)
                }
            }
        }

        return kept
    }

    /// Compute Intersection over Union between two boxes
    private func iou(_ a: DetectorBox, _ b: DetectorBox) -> Float {
        let interX0 = max(a.x0, b.x0)
        let interY0 = max(a.y0, b.y0)
        let interX1 = min(a.x1, b.x1)
        let interY1 = min(a.y1, b.y1)

        let interArea = max(0, interX1 - interX0) * max(0, interY1 - interY0)
        let areaA = (a.x1 - a.x0) * (a.y1 - a.y0)
        let areaB = (b.x1 - b.x0) * (b.y1 - b.y0)
        let unionArea = areaA + areaB - interArea

        return unionArea > 0 ? interArea / unionArea : 0
    }
}

// MARK: - Supporting Types

/// Transform parameters for YOLO-World coordinate conversion
public struct GroundingTransform {
    public let scale: Float
    public let padX: Float
    public let padY: Float
    public let originalWidth: Int
    public let originalHeight: Int
    public let inputSize: Int
}

/// Raw bounding box from detector (xyxy format)
public struct DetectorBox {
    public let x0: Float
    public let y0: Float
    public let x1: Float
    public let y1: Float
}
