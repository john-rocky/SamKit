import Foundation
import CoreML
import CoreGraphics
import Accelerate

/// Main session class for SAM inference on iOS
public final class SamSession {
    
    // MARK: - Properties

    private let encoder: MLModel
    private let decoder: MLModel
    private let config: RuntimeConfig
    private let preprocessor: Preprocessor
    private let postprocessor: Postprocessor
    private let promptEncoder: PromptEncoder?
    private let modelType: ModelType

    private var cachedEmbedding: MLMultiArray?
    private var transformParams: TransformParams?
    private let modelSize: Int

    // MARK: - Initialization

    public init(model: SamModelRef, config: RuntimeConfig = .bestAvailable) throws {
        self.config = config
        self.modelSize = model.inputSize
        self.modelType = model.modelType

        // Load Core ML models with optimized configuration
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits.mlComputeUnits

        // Enable low precision computation for better performance
        mlConfig.allowLowPrecisionAccumulationOnGPU = true

        self.encoder = try MLModel(contentsOf: model.encoderURL, configuration: mlConfig)
        self.decoder = try MLModel(contentsOf: model.decoderURL, configuration: mlConfig)

        // Initialize processors
        self.preprocessor = Preprocessor(modelSize: modelSize)
        self.postprocessor = Postprocessor(
            modelSize: modelSize,
            isHuggingFaceModel: model.modelType != .mobileSam
        )

        // Load prompt encoder weights for MobileSAM (decoder does not include prompt encoding)
        if model.modelType == .mobileSam,
           let weightsURL = model.promptEncoderWeightsURL {
            self.promptEncoder = try PromptEncoder(weightsURL: weightsURL)
        } else {
            self.promptEncoder = nil
        }
    }
    
    // MARK: - Public Methods
    
    /// Set the image for segmentation
    public func setImage(_ image: CGImage) throws {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Clear previous cache
        cachedEmbedding = nil
        transformParams = nil

        // Preprocess image
        let (processedImage, transform) = try preprocessor.process(image)
        self.transformParams = transform

        let t1 = CFAbsoluteTimeGetCurrent()

        // Add batch dimension [3, H, W] → [1, 3, H, W] for models that expect it
        let batchedImage: MLMultiArray
        if processedImage.shape.count == 3 {
            let s = processedImage.shape.map { $0.intValue }
            batchedImage = try MLMultiArray(shape: [1, s[0] as NSNumber, s[1] as NSNumber, s[2] as NSNumber], dataType: .float32)
            let src = processedImage.dataPointer.bindMemory(to: Float32.self, capacity: processedImage.count)
            let dst = batchedImage.dataPointer.bindMemory(to: Float32.self, capacity: batchedImage.count)
            memcpy(dst, src, processedImage.count * MemoryLayout<Float32>.size)
        } else {
            batchedImage = processedImage
        }

        // Run encoder
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "image": batchedImage
        ])

        let encoderOutput = try encoder.prediction(from: encoderInput)

        let t2 = CFAbsoluteTimeGetCurrent()

        // Cache embedding
        guard let embedding = encoderOutput.featureValue(for: "image_embeddings")?.multiArrayValue else {
            throw SamError.invalidModelOutput("Missing image_embeddings from encoder")
        }

        self.cachedEmbedding = embedding
        print("[SAMKit] setImage: preprocess=\(Int((t1-t0)*1000))ms encoder=\(Int((t2-t1)*1000))ms")
    }
    
    /// Run mask prediction with prompts
    public func predict(
        points: [SamPoint] = [],
        box: SamBox? = nil,
        maskInput: SamMaskRef? = nil,
        options: SamOptions = SamOptions()
    ) throws -> SamResult {

        let t0 = CFAbsoluteTimeGetCurrent()

        guard let embedding = cachedEmbedding,
              let transform = transformParams else {
            throw SamError.imageNotSet
        }

        // Build decoder inputs depending on model type
        let decoderInput: MLDictionaryFeatureProvider

        if let promptEnc = promptEncoder {
            let (sparseEmb, denseEmb) = try promptEnc.encode(points: points, box: box, transform: transform)
            decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "image_embeddings": embedding,
                "sparse_embeddings": sparseEmb,
                "dense_embeddings": denseEmb,
            ])
        } else {
            // Build combined points: user points + box corners (as label 2, 3)
            var allPoints = points
            if let box = box {
                allPoints.append(SamPoint(x: CGFloat(box.x0), y: CGFloat(box.y0), label: .positive))
                allPoints.append(SamPoint(x: CGFloat(box.x1), y: CGFloat(box.y1), label: .positive))
            }

            let (pointCoords, pointLabels) = try preprocessor.encodePoints(allPoints, transform: transform)

            // Override labels for box corners to SAM convention (2 = top-left, 3 = bottom-right)
            if box != nil {
                let labelsPtr = pointLabels.dataPointer.bindMemory(to: Float32.self, capacity: pointLabels.count)
                let boxStart = allPoints.count - 2
                labelsPtr[boxStart] = 2.0
                labelsPtr[boxStart + 1] = 3.0
            }

            var inputs: [String: Any] = [
                "image_embeddings": embedding,
                "point_coords": pointCoords,
                "point_labels": pointLabels,
                "has_mask_input": MLMultiArray.scalar(Float(maskInput != nil ? 1.0 : 0.0))
            ]
            if let maskInput = maskInput {
                inputs["mask_input"] = try preprocessor.encodeMask(maskInput, transform: transform)
            } else {
                inputs["mask_input"] = try MLMultiArray.zeros(shape: [1, 1, 256, 256])
            }
            decoderInput = try MLDictionaryFeatureProvider(dictionary: inputs)
        }

        let t1 = CFAbsoluteTimeGetCurrent()

        // Run decoder
        let decoderOutput = try decoder.prediction(from: decoderInput)

        let t2 = CFAbsoluteTimeGetCurrent()

        // Extract masks and scores
        guard let maskLogits = decoderOutput.featureValue(for: "masks")?.multiArrayValue,
              let iouPredictions = decoderOutput.featureValue(for: "iou_predictions")?.multiArrayValue else {
            throw SamError.invalidModelOutput("Missing masks or iou_predictions from decoder")
        }

        // Postprocess results
        let result = try postprocessor.process(
            maskLogits: maskLogits,
            iouPredictions: iouPredictions,
            transform: transform,
            options: options
        )

        let t3 = CFAbsoluteTimeGetCurrent()
        print("[SAMKit] prompt=\(Int((t1-t0)*1000))ms decoder=\(Int((t2-t1)*1000))ms postprocess=\(Int((t3-t2)*1000))ms total=\(Int((t3-t0)*1000))ms")

        return result
    }
    
    /// Clear cached embedding
    public func clear() {
        cachedEmbedding = nil
        transformParams = nil
    }
}

// MARK: - Supporting Types

public struct SamModelRef {
    public let encoderURL: URL
    public let decoderURL: URL
    public let inputSize: Int
    public let modelType: ModelType
    public let promptEncoderWeightsURL: URL?

    public init(
        encoderURL: URL,
        decoderURL: URL,
        inputSize: Int = 1024,
        modelType: ModelType,
        promptEncoderWeightsURL: URL? = nil
    ) {
        self.encoderURL = encoderURL
        self.decoderURL = decoderURL
        self.inputSize = inputSize
        self.modelType = modelType
        self.promptEncoderWeightsURL = promptEncoderWeightsURL
    }

    /// Load model from bundle
    public static func bundled(_ modelType: ModelType) throws -> SamModelRef {
        let bundle = Bundle.main
        let modelName = modelType.modelName

        guard let encoderURL = bundle.url(forResource: "\(modelName)_encoder", withExtension: "mlmodelc"),
              let decoderURL = bundle.url(forResource: "\(modelName)_decoder", withExtension: "mlmodelc") else {
            throw SamError.modelNotFound
        }

        // Look for prompt encoder weights JSON (needed for MobileSAM)
        let weightsURL = bundle.url(
            forResource: "\(modelName)_prompt_encoder_weights",
            withExtension: "json"
        )

        return SamModelRef(
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            inputSize: modelType.inputSize,
            modelType: modelType,
            promptEncoderWeightsURL: weightsURL
        )
    }
}

public enum ModelType {
    case sam2_1_tiny
    case sam2_1_small
    case sam2_1_base
    case sam2_1_large
    case sam2_1_basePlus
    case mobileSam
    
    public var modelName: String {
        switch self {
        case .sam2_1_tiny: return "sam2_tiny"
        case .sam2_1_small: return "sam2_small"
        case .sam2_1_base: return "sam2_base"
        case .sam2_1_large: return "sam2_large"
        case .sam2_1_basePlus: return "sam2_base_plus"
        case .mobileSam: return "mobile_sam"
        }
    }
    
    public var inputSize: Int {
        return 1024  // All models use 1024x1024 input
    }
}

public struct RuntimeConfig {
    public enum ComputeUnits {
        case cpuOnly
        case gpuPreferred
        case neuralEnginePreferred
        case bestAvailable
        
        public var mlComputeUnits: MLComputeUnits {
            switch self {
            case .cpuOnly: return .cpuOnly
            case .gpuPreferred: return .cpuAndGPU
            case .neuralEnginePreferred: 
                // Prefer Neural Engine for best performance
                if #available(iOS 16.0, *) {
                    return .cpuAndNeuralEngine
                } else {
                    return .cpuAndGPU
                }
            case .bestAvailable: 
                // Use Neural Engine + GPU for optimal performance
                return .all
            }
        }
    }
    
    public let computeUnits: ComputeUnits
    public let enableFP16: Bool
    
    public init(computeUnits: ComputeUnits = .bestAvailable, enableFP16: Bool = true) {
        self.computeUnits = computeUnits
        self.enableFP16 = enableFP16
    }
    
    public static let bestAvailable = RuntimeConfig()
}

public struct SamPoint {
    public let x: CGFloat
    public let y: CGFloat
    public let label: SamPointLabel
    
    public init(x: CGFloat, y: CGFloat, label: SamPointLabel) {
        self.x = x
        self.y = y
        self.label = label
    }
}

public enum SamPointLabel: Int {
    case positive = 1
    case negative = 0
}

public struct SamBox {
    public let x0: Float
    public let y0: Float
    public let x1: Float
    public let y1: Float
    
    public init(x0: Float, y0: Float, x1: Float, y1: Float) {
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    }
}

public struct SamOptions {
    public let multimaskOutput: Bool
    public let returnLogits: Bool
    public let maskThreshold: Float
    public let maxMasks: Int
    
    public init(
        multimaskOutput: Bool = true,
        returnLogits: Bool = false,
        maskThreshold: Float = 0.0,
        maxMasks: Int = 3
    ) {
        self.multimaskOutput = multimaskOutput
        self.returnLogits = returnLogits
        self.maskThreshold = maskThreshold
        self.maxMasks = maxMasks
    }
}

public struct SamMask {
    public let width: Int
    public let height: Int
    public let logits: [Float]?
    public let alpha: Data
    public let score: Float
    public let cgImage: CGImage
    
    public init(width: Int, height: Int, logits: [Float]?, alpha: Data, score: Float, cgImage: CGImage) {
        self.width = width
        self.height = height
        self.logits = logits
        self.alpha = alpha
        self.score = score
        self.cgImage = cgImage
    }
}

public typealias SamMaskRef = SamMask

public struct SamResult {
    public let masks: [SamMask]
    public let scores: [Float]
    
    public init(masks: [SamMask], scores: [Float]) {
        self.masks = masks
        self.scores = scores
    }
}

public enum SamError: LocalizedError {
    case imageNotSet
    case modelNotFound
    case invalidModelOutput(String)
    case preprocessingFailed(String)
    case postprocessingFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .imageNotSet:
            return "Image not set. Call setImage() first."
        case .modelNotFound:
            return "Model files not found in bundle"
        case .invalidModelOutput(let message):
            return "Invalid model output: \(message)"
        case .preprocessingFailed(let message):
            return "Preprocessing failed: \(message)"
        case .postprocessingFailed(let message):
            return "Postprocessing failed: \(message)"
        }
    }
}

// MARK: - MLMultiArray Extensions

extension MLMultiArray {
    static func zeros(shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        for i in 0..<array.count {
            ptr[i] = 0
        }
        return array
    }
    
    static func scalar(_ value: Float) -> MLMultiArray {
        if let array = try? MLMultiArray(shape: [1], dataType: .float32) {
            array[0] = NSNumber(value: value)
            return array
        }
        fatalError("Failed to create scalar MLMultiArray")
    }
}