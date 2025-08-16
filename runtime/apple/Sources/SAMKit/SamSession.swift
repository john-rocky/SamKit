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
    
    private var cachedEmbedding: MLMultiArray?
    private var transformParams: TransformParams?
    private let modelSize: Int
    
    // MARK: - Initialization
    
    public init(model: SamModelRef, config: RuntimeConfig = .bestAvailable) throws {
        self.config = config
        self.modelSize = model.inputSize
        
        // Load Core ML models with optimized configuration
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits.mlComputeUnits
        
        // Enable low precision computation for better performance
        mlConfig.allowLowPrecisionAccumulationOnGPU = true
        
        // Set prediction options for better performance
        // Note: optimizationHints property doesn't have .performance, using default
        
        self.encoder = try MLModel(contentsOf: model.encoderURL, configuration: mlConfig)
        self.decoder = try MLModel(contentsOf: model.decoderURL, configuration: mlConfig)
        
        // Initialize processors
        self.preprocessor = Preprocessor(modelSize: modelSize)
        self.postprocessor = Postprocessor(modelSize: modelSize)
    }
    
    // MARK: - Public Methods
    
    /// Set the image for segmentation
    public func setImage(_ image: CGImage) throws {
        // Clear previous cache
        cachedEmbedding = nil
        transformParams = nil
        
        // Preprocess image
        let (processedImage, transform) = try preprocessor.process(image)
        self.transformParams = transform
        
        // Run encoder
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "image": processedImage
        ])
        
        let encoderOutput = try encoder.prediction(from: encoderInput)
        
        // Cache embedding
        guard let embedding = encoderOutput.featureValue(for: "image_embeddings")?.multiArrayValue else {
            throw SamError.invalidModelOutput("Missing image_embeddings from encoder")
        }
        
        self.cachedEmbedding = embedding
    }
    
    /// Run mask prediction with prompts
    public func predict(
        points: [SamPoint] = [],
        box: SamBox? = nil,
        maskInput: SamMaskRef? = nil,
        options: SamOptions = SamOptions()
    ) async throws -> SamResult {
        
        guard let embedding = cachedEmbedding,
              let transform = transformParams else {
            throw SamError.imageNotSet
        }
        
        // Encode prompts
        let (pointCoords, pointLabels) = try preprocessor.encodePoints(points, transform: transform)
        let hasMaskInput = maskInput != nil ? 1.0 : 0.0
        
        // Prepare decoder inputs
        var decoderInputs: [String: Any] = [
            "image_embeddings": embedding,
            "point_coords": pointCoords,
            "point_labels": pointLabels,
            "has_mask_input": MLMultiArray.scalar(Float(hasMaskInput))
        ]
        
        // Add mask input if provided
        if let maskInput = maskInput {
            let encodedMask = try preprocessor.encodeMask(maskInput, transform: transform)
            decoderInputs["mask_input"] = encodedMask
        } else {
            // Provide empty mask
            decoderInputs["mask_input"] = try MLMultiArray.zeros(shape: [1, 1, 256, 256])
        }
        
        // Run decoder
        let decoderInput = try MLDictionaryFeatureProvider(dictionary: decoderInputs)
        let decoderOutput = try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    let output = try decoder.prediction(from: decoderInput)
                    continuation.resume(returning: output)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        
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
    
    public init(encoderURL: URL, decoderURL: URL, inputSize: Int = 1024, modelType: ModelType) {
        self.encoderURL = encoderURL
        self.decoderURL = decoderURL
        self.inputSize = inputSize
        self.modelType = modelType
    }
    
    /// Load model from bundle
    public static func bundled(_ modelType: ModelType) throws -> SamModelRef {
        let bundle = Bundle.main
        
        let modelName = modelType.modelName
        
        guard let encoderURL = bundle.url(forResource: "\(modelName)_encoder", withExtension: "mlmodelc"),
              let decoderURL = bundle.url(forResource: "\(modelName)_decoder", withExtension: "mlmodelc") else {
            throw SamError.modelNotFound
        }
        
        return SamModelRef(
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            inputSize: modelType.inputSize,
            modelType: modelType
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
        
        var mlComputeUnits: MLComputeUnits {
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