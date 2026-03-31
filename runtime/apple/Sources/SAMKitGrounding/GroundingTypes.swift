import Foundation
import CoreML
import CoreGraphics
import SAMKit

// MARK: - Model Reference

/// Reference to YOLO-World grounding model files
public struct GroundingModelRef {
    public let textEncoderURL: URL
    public let detectorURL: URL
    public let vocabularyURL: URL
    public let inputSize: Int
    public let maxClasses: Int
    public let contextLength: Int

    public init(
        textEncoderURL: URL,
        detectorURL: URL,
        vocabularyURL: URL,
        inputSize: Int = 640,
        maxClasses: Int = 80,
        contextLength: Int = 77
    ) {
        self.textEncoderURL = textEncoderURL
        self.detectorURL = detectorURL
        self.vocabularyURL = vocabularyURL
        self.inputSize = inputSize
        self.maxClasses = maxClasses
        self.contextLength = contextLength
    }

    /// Load grounding models from app bundle
    public static func bundled(
        textEncoderName: String = "clip_text_encoder",
        detectorName: String = "yoloworld_detector",
        vocabName: String = "clip_vocab"
    ) throws -> GroundingModelRef {
        let bundle = Bundle.main

        guard let textEncoderURL = bundle.url(forResource: textEncoderName, withExtension: "mlmodelc")
                ?? bundle.url(forResource: textEncoderName, withExtension: "mlpackage") else {
            throw GroundingError.modelNotFound("Text encoder '\(textEncoderName)' not found")
        }

        guard let detectorURL = bundle.url(forResource: detectorName, withExtension: "mlmodelc")
                ?? bundle.url(forResource: detectorName, withExtension: "mlpackage") else {
            throw GroundingError.modelNotFound("Detector '\(detectorName)' not found")
        }

        guard let vocabURL = bundle.url(forResource: vocabName, withExtension: "json") else {
            throw GroundingError.modelNotFound("Vocabulary '\(vocabName).json' not found")
        }

        return GroundingModelRef(
            textEncoderURL: textEncoderURL,
            detectorURL: detectorURL,
            vocabularyURL: vocabURL
        )
    }
}

// MARK: - Detection Result

/// A single object detection from the grounding model
public struct GroundingResult {
    /// Bounding box in original image coordinates
    public let box: SamBox
    /// Detection confidence score (0-1)
    public let confidence: Float
    /// Matched class label
    public let label: String

    public init(box: SamBox, confidence: Float, label: String) {
        self.box = box
        self.confidence = confidence
        self.label = label
    }
}

// MARK: - Options

/// Options for text-prompted segmentation
public struct TextPromptOptions {
    /// Minimum confidence threshold for detections
    public let confidenceThreshold: Float
    /// IoU threshold for Non-Maximum Suppression
    public let nmsThreshold: Float
    /// Maximum number of detections to return
    public let maxDetections: Int
    /// Whether SAM should return multiple mask candidates per detection
    public let multimaskOutput: Bool

    public init(
        confidenceThreshold: Float = 0.01,
        nmsThreshold: Float = 0.5,
        maxDetections: Int = 5,
        multimaskOutput: Bool = false
    ) {
        self.confidenceThreshold = confidenceThreshold
        self.nmsThreshold = nmsThreshold
        self.maxDetections = maxDetections
        self.multimaskOutput = multimaskOutput
    }
}

// MARK: - Text Segmentation Result

/// Combined result of text-prompted detection + segmentation
public struct TextSegmentationResult {
    /// Detected objects with bounding boxes
    public let detections: [GroundingResult]
    /// Segmentation masks (one per detection)
    public let masks: [SamMask]
    /// SAM IoU confidence scores
    public let scores: [Float]

    public init(detections: [GroundingResult], masks: [SamMask], scores: [Float]) {
        self.detections = detections
        self.masks = masks
        self.scores = scores
    }
}

// MARK: - Errors

public enum GroundingError: LocalizedError {
    case modelNotFound(String)
    case imageNotSet
    case invalidModelOutput(String)
    case tokenizationFailed(String)
    case detectionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return "Model not found: \(msg)"
        case .imageNotSet: return "Image not set. Call setImage() first."
        case .invalidModelOutput(let msg): return "Invalid model output: \(msg)"
        case .tokenizationFailed(let msg): return "Tokenization failed: \(msg)"
        case .detectionFailed(let msg): return "Detection failed: \(msg)"
        }
    }
}
