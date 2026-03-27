import Foundation
import CoreML
import CoreGraphics
import Accelerate

/// SAM2 model support with separate PromptEncoder
public final class Sam2Session {
    
    // MARK: - Properties
    
    private let imageEncoder: MLModel
    private let promptEncoder: MLModel
    private let maskDecoder: MLModel
    private let config: RuntimeConfig
    private let preprocessor: Preprocessor
    private let postprocessor: Postprocessor
    
    private var cachedEmbedding: MLMultiArray?
    private var cachedEncoderOutputs: [String: MLMultiArray]?
    private var transformParams: TransformParams?
    private let modelSize: Int
    
    // MARK: - Initialization
    
    public init(
        imageEncoderURL: URL,
        promptEncoderURL: URL,
        maskDecoderURL: URL,
        modelSize: Int = 1024,
        config: RuntimeConfig = .bestAvailable
    ) throws {
        self.config = config
        self.modelSize = modelSize
        
        // Load Core ML models with optimized configuration
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits.mlComputeUnits
        
        // Enable low precision computation for better performance
        mlConfig.allowLowPrecisionAccumulationOnGPU = true
        
        // Set prediction options for better performance
        // Note: optimizationHints property doesn't have .performance, using default
        
        self.imageEncoder = try MLModel(contentsOf: imageEncoderURL, configuration: mlConfig)
        self.promptEncoder = try MLModel(contentsOf: promptEncoderURL, configuration: mlConfig)
        self.maskDecoder = try MLModel(contentsOf: maskDecoderURL, configuration: mlConfig)
        
        // Initialize processors
        // Note: isHuggingFaceModel must be false — SAM2 outputs [1,3,256,256] as 3 separate masks,
        // NOT the 2x2 grid layout that isHuggingFaceModel triggers
        self.preprocessor = Preprocessor(modelSize: modelSize)
        self.postprocessor = Postprocessor(modelSize: modelSize, isHuggingFaceModel: false)
    }
    
    /// Convenience initializer for HuggingFace models
    public convenience init(modelName: String = "SAM2Tiny", config: RuntimeConfig = .bestAvailable) throws {
        let bundle = Bundle.main
        
        // Look for the three separate model files (compiled to .mlmodelc)
        guard let imageEncoderURL = bundle.url(forResource: "\(modelName)ImageEncoderFLOAT16", withExtension: "mlmodelc"),
              let promptEncoderURL = bundle.url(forResource: "\(modelName)PromptEncoderFLOAT16", withExtension: "mlmodelc"),
              let maskDecoderURL = bundle.url(forResource: "\(modelName)MaskDecoderFLOAT16", withExtension: "mlmodelc") else {
            throw SamError.modelNotFound
        }
        
        try self.init(
            imageEncoderURL: imageEncoderURL,
            promptEncoderURL: promptEncoderURL,
            maskDecoderURL: maskDecoderURL,
            config: config
        )
    }
    
    // MARK: - Public Methods
    
    /// Set the image for segmentation
    public func setImage(_ image: CGImage) throws {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Clear previous cache
        cachedEmbedding = nil
        cachedEncoderOutputs = nil
        transformParams = nil

        // Preprocess image for SAM2 (returns CGImage)
        let (processedImage, transform) = try preprocessor.processForSAM2(image)
        self.transformParams = transform

        // Convert CGImage to MLFeatureValue
        let imageFeature = try preprocessor.createImageFeature(processedImage)

        let t1 = CFAbsoluteTimeGetCurrent()

        // Run image encoder
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "image": imageFeature
        ])

        let encoderOutput = try imageEncoder.prediction(from: encoderInput)

        let t2 = CFAbsoluteTimeGetCurrent()

        // Cache all encoder outputs - HuggingFace SAM2 needs multiple features
        var allEncoderOutputs: [String: MLMultiArray] = [:]

        for featureName in encoderOutput.featureNames {
            if let featureValue = encoderOutput.featureValue(for: featureName)?.multiArrayValue {
                allEncoderOutputs[featureName] = featureValue
            }
        }

        // Find the main embedding
        let embedding: MLMultiArray?
        if let emb = allEncoderOutputs["image_embedding"] {
            embedding = emb
        } else if let emb = allEncoderOutputs["image_embeddings"] {
            embedding = emb
        } else {
            embedding = allEncoderOutputs.values.first
        }

        guard let finalEmbedding = embedding else {
            throw SamError.invalidModelOutput("Could not extract image embeddings")
        }

        self.cachedEmbedding = finalEmbedding
        self.cachedEncoderOutputs = allEncoderOutputs

        print("[SAM2] setImage: preprocess=\(Int((t1-t0)*1000))ms encoder=\(Int((t2-t1)*1000))ms")
    }
    
    /// Run mask prediction with prompts
    public func predict(
        points: [SamPoint] = [],
        box: SamBox? = nil,
        maskInput: SamMaskRef? = nil,
        options: SamOptions = SamOptions()
    ) throws -> SamResult {

        let t0 = CFAbsoluteTimeGetCurrent()

        guard let _ = cachedEmbedding,
              let encoderOutputs = cachedEncoderOutputs,
              let transform = transformParams else {
            throw SamError.imageNotSet
        }

        // Encode prompts using the separate PromptEncoder
        let promptEmbeddings = try encodePrompts(
            points: points,
            box: box,
            maskInput: maskInput,
            transform: transform
        )

        let t1 = CFAbsoluteTimeGetCurrent()

        // Prepare mask decoder inputs
        var decoderInputs: [String: Any] = [:]

        // Add all encoder outputs (includes feats_s0, feats_s1, image_embedding)
        for (key, value) in encoderOutputs {
            decoderInputs[key] = value
        }

        // Add prompt embeddings (provide both singular/plural keys for model compatibility)
        if let sparseEmbeddings = promptEmbeddings["sparse_embeddings"] {
            decoderInputs["sparse_embeddings"] = sparseEmbeddings
            decoderInputs["sparse_embedding"] = sparseEmbeddings
        }
        if let denseEmbeddings = promptEmbeddings["dense_embeddings"] {
            decoderInputs["dense_embeddings"] = denseEmbeddings
            decoderInputs["dense_embedding"] = denseEmbeddings
        }

        // Run mask decoder directly (no Task.detached overhead)
        let decoderInput = try MLDictionaryFeatureProvider(dictionary: decoderInputs)
        let decoderOutput = try maskDecoder.prediction(from: decoderInput)

        let t2 = CFAbsoluteTimeGetCurrent()

        // Extract masks and scores
        let maskLogits = decoderOutput.featureValue(for: "masks")?.multiArrayValue
            ?? decoderOutput.featureValue(for: "low_res_masks")?.multiArrayValue
            ?? decoderOutput.featureValue(for: "output_0")?.multiArrayValue

        let iouPredictions = decoderOutput.featureValue(for: "iou_predictions")?.multiArrayValue
            ?? decoderOutput.featureValue(for: "scores")?.multiArrayValue
            ?? decoderOutput.featureValue(for: "output_1")?.multiArrayValue

        guard let finalMaskLogits = maskLogits,
              let finalIouPredictions = iouPredictions else {
            let keys = decoderOutput.featureNames.joined(separator: ", ")
            throw SamError.invalidModelOutput("Missing masks or iou_predictions from decoder. Available keys: \(keys)")
        }

        // Postprocess results
        let result = try postprocessor.process(
            maskLogits: finalMaskLogits,
            iouPredictions: finalIouPredictions,
            transform: transform,
            options: options
        )

        let t3 = CFAbsoluteTimeGetCurrent()
        print("[SAM2] prompt=\(Int((t1-t0)*1000))ms decoder=\(Int((t2-t1)*1000))ms postprocess=\(Int((t3-t2)*1000))ms total=\(Int((t3-t0)*1000))ms")

        return result
    }
    
    /// Clear cached embedding
    public func clear() {
        cachedEmbedding = nil
        cachedEncoderOutputs = nil
        transformParams = nil
    }
    
    // MARK: - Private Methods
    
    private func encodePrompts(
        points: [SamPoint],
        box: SamBox?,
        maskInput: SamMaskRef?,
        transform: TransformParams
    ) throws -> [String: MLMultiArray] {
        
        var promptInputs: [String: Any] = [:]
        
        // Encode points if provided
        if !points.isEmpty {
            let (pointCoords, pointLabels) = try preprocessor.encodePoints(points, transform: transform)
            promptInputs["point_coords"] = pointCoords
            promptInputs["point_labels"] = pointLabels
            // HuggingFace models may use "points" as a combined input
            promptInputs["points"] = pointCoords
            promptInputs["labels"] = pointLabels
        } else {
            // Provide default point at center (SAM2 may require at least 1 point)
            let defaultCoords = try MLMultiArray(shape: [1, 1, 2], dataType: .float32)
            defaultCoords[[0, 0, 0] as [NSNumber]] = NSNumber(value: 512.0)  // Center X
            defaultCoords[[0, 0, 1] as [NSNumber]] = NSNumber(value: 512.0)  // Center Y
            
            let defaultLabels = try MLMultiArray(shape: [1, 1], dataType: .float32)
            defaultLabels[[0, 0] as [NSNumber]] = NSNumber(value: 1.0)  // Positive label
            
            promptInputs["point_coords"] = defaultCoords
            promptInputs["point_labels"] = defaultLabels
            // HuggingFace models may use "points" as a combined input
            promptInputs["points"] = defaultCoords
            promptInputs["labels"] = defaultLabels
        }
        
        // Encode box if provided
        if let box = box {
            let encodedBox = try preprocessor.encodeBox(box, transform: transform)
            promptInputs["boxes"] = encodedBox
        } else {
            // Provide zero box to signal "no box prompt"
            let emptyBox = try MLMultiArray(shape: [1, 1, 4], dataType: .float32)
            promptInputs["boxes"] = emptyBox
        }
        
        // Encode mask if provided
        if let maskInput = maskInput {
            let encodedMask = try preprocessor.encodeMask(maskInput, transform: transform)
            promptInputs["mask_inputs"] = encodedMask
            promptInputs["masks"] = encodedMask  // HuggingFace may use "masks"
        } else {
            // Provide zero mask with proper shape
            let zeroMask = try MLMultiArray.zeros(shape: [1, 1, 256, 256])
            promptInputs["mask_inputs"] = zeroMask
            promptInputs["masks"] = zeroMask  // HuggingFace may use "masks"
        }
        
        // Add has_mask_input flag (some models may need this)
        promptInputs["has_mask_input"] = MLMultiArray.scalar(maskInput != nil ? 1.0 : 0.0)
        
        // Run prompt encoder
        let promptInput: MLDictionaryFeatureProvider
        do {
            promptInput = try MLDictionaryFeatureProvider(dictionary: promptInputs)
        } catch {
            // If we get an error, try to understand what inputs the model expects
            print("Error creating prompt input: \(error)")
            print("Provided keys: \(promptInputs.keys.joined(separator: ", "))")
            throw error
        }
        
        let promptOutput: MLFeatureProvider
        do {
            promptOutput = try promptEncoder.prediction(from: promptInput)
        } catch {
            // Debug: show what the model expects
            if let modelError = error as NSError?,
               modelError.domain == "com.apple.CoreML" {
                print("PromptEncoder error: \(modelError.localizedDescription)")
                print("Provided inputs: \(promptInputs.keys.joined(separator: ", "))")
            }
            throw error
        }
        
        // Extract embeddings - try different possible output keys
        var embeddings: [String: MLMultiArray] = [:]
        
        // Try common output keys for sparse embeddings
        if let sparseEmbeddings = promptOutput.featureValue(for: "sparse_embeddings")?.multiArrayValue {
            embeddings["sparse_embeddings"] = sparseEmbeddings
        } else if let sparseEmbeddings = promptOutput.featureValue(for: "output_0")?.multiArrayValue {
            embeddings["sparse_embeddings"] = sparseEmbeddings
        } else if let sparseEmbeddings = promptOutput.featureValue(for: "var_6995")?.multiArrayValue {
            // Some HuggingFace models use auto-generated names
            embeddings["sparse_embeddings"] = sparseEmbeddings
        }
        
        // Try common output keys for dense embeddings
        if let denseEmbeddings = promptOutput.featureValue(for: "dense_embeddings")?.multiArrayValue {
            embeddings["dense_embeddings"] = denseEmbeddings
        } else if let denseEmbeddings = promptOutput.featureValue(for: "output_1")?.multiArrayValue {
            embeddings["dense_embeddings"] = denseEmbeddings
        } else if let denseEmbeddings = promptOutput.featureValue(for: "var_6996")?.multiArrayValue {
            // Some HuggingFace models use auto-generated names
            embeddings["dense_embeddings"] = denseEmbeddings
        }
        
        // If no embeddings found, provide debug info
        if embeddings.isEmpty {
            let keys = promptOutput.featureNames.joined(separator: ", ")
            print("Warning: No prompt embeddings found. Available keys: \(keys)")
        }
        
        return embeddings
    }
}

// MARK: - SAM2 Model Reference

public struct Sam2ModelRef {
    public let imageEncoderURL: URL
    public let promptEncoderURL: URL
    public let maskDecoderURL: URL
    public let inputSize: Int
    public let modelType: Sam2ModelType
    
    public init(
        imageEncoderURL: URL,
        promptEncoderURL: URL,
        maskDecoderURL: URL,
        inputSize: Int = 1024,
        modelType: Sam2ModelType
    ) {
        self.imageEncoderURL = imageEncoderURL
        self.promptEncoderURL = promptEncoderURL
        self.maskDecoderURL = maskDecoderURL
        self.inputSize = inputSize
        self.modelType = modelType
    }
    
    /// Load SAM2 model from bundle (HuggingFace format)
    public static func fromHuggingFace(modelType: Sam2ModelType = .tiny) throws -> Sam2ModelRef {
        let bundle = Bundle.main
        
        let modelPrefix = modelType.modelPrefix
        
        guard let imageEncoderURL = bundle.url(
                forResource: "\(modelPrefix)ImageEncoderFLOAT16",
                withExtension: "mlmodelc"
              ),
              let promptEncoderURL = bundle.url(
                forResource: "\(modelPrefix)PromptEncoderFLOAT16",
                withExtension: "mlmodelc"
              ),
              let maskDecoderURL = bundle.url(
                forResource: "\(modelPrefix)MaskDecoderFLOAT16",
                withExtension: "mlmodelc"
              ) else {
            throw SamError.modelNotFound
        }
        
        return Sam2ModelRef(
            imageEncoderURL: imageEncoderURL,
            promptEncoderURL: promptEncoderURL,
            maskDecoderURL: maskDecoderURL,
            inputSize: modelType.inputSize,
            modelType: modelType
        )
    }
}

public enum Sam2ModelType {
    case tiny

    public var modelPrefix: String {
        return "SAM2Tiny"
    }

    public var inputSize: Int {
        return 1024
    }

    public var displayName: String {
        return "SAM 2 Tiny"
    }
}