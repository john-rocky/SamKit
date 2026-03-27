import Foundation
import CoreML
import CoreGraphics
import SAMKit

/// High-level session that combines YOLO-World text detection with SAM segmentation.
///
/// Usage:
/// ```swift
/// let session = try TextSegmentationSession(
///     groundingModel: .bundled(),
///     samModel: .bundled(.mobileSam)
/// )
/// try session.setImage(cgImage)
/// let result = try await session.segment(query: "dog")
/// ```
public final class TextSegmentationSession {

    // MARK: - Properties

    private let detector: TextDetector
    private let samSession: SamSession
    private var currentImage: CGImage?

    // MARK: - Initialization

    public init(
        groundingModel: GroundingModelRef,
        samModel: SamModelRef,
        config: RuntimeConfig = .bestAvailable
    ) throws {
        self.detector = try TextDetector(model: groundingModel, config: config)
        self.samSession = try SamSession(model: samModel, config: config)
    }

    // MARK: - Public Methods

    /// Set the image for both detection and segmentation.
    /// Runs the SAM image encoder (cached for subsequent queries).
    public func setImage(_ image: CGImage) throws {
        let t0 = CFAbsoluteTimeGetCurrent()

        self.currentImage = image
        try samSession.setImage(image)

        let t1 = CFAbsoluteTimeGetCurrent()
        print("[TextSegmentation] setImage: \(Int((t1-t0)*1000))ms")
    }

    /// Segment objects matching the text query.
    ///
    /// The query can contain multiple object names separated by commas.
    /// For example: "dog, cat" will detect dogs and cats separately.
    public func segment(
        query: String,
        options: TextPromptOptions = TextPromptOptions()
    ) throws -> TextSegmentationResult {
        guard let image = currentImage else {
            throw GroundingError.imageNotSet
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Parse query into individual class names
        let queries = query
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        guard !queries.isEmpty else {
            return TextSegmentationResult(detections: [], masks: [], scores: [])
        }

        // Detect objects
        let detections = try detector.detect(
            image: image,
            queries: queries,
            options: options
        )

        let t1 = CFAbsoluteTimeGetCurrent()

        guard !detections.isEmpty else {
            print("[TextSegmentation] No objects detected for query: \(query)")
            return TextSegmentationResult(detections: [], masks: [], scores: [])
        }

        // Run SAM segmentation for each detected box
        var masks: [SamMask] = []
        var scores: [Float] = []

        let samOptions = SamOptions(
            multimaskOutput: options.multimaskOutput,
            returnLogits: false,
            maskThreshold: 0.0,
            maxMasks: 1
        )

        for detection in detections {
            let result = try samSession.predict(
                points: [],
                box: detection.box,
                options: samOptions
            )

            if let bestMask = result.masks.first {
                masks.append(bestMask)
                scores.append(result.scores.first ?? 0)
            }
        }

        let t2 = CFAbsoluteTimeGetCurrent()
        print("[TextSegmentation] detect=\(Int((t1-t0)*1000))ms segment=\(Int((t2-t1)*1000))ms (\(detections.count) objects) total=\(Int((t2-t0)*1000))ms")

        return TextSegmentationResult(
            detections: detections,
            masks: masks,
            scores: scores
        )
    }

    /// Clear cached state
    public func clear() {
        currentImage = nil
        samSession.clear()
        detector.clearCache()
    }
}
