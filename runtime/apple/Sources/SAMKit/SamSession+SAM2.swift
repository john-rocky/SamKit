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
        self.preprocessor = Preprocessor(modelSize: modelSize)
        self.postprocessor = Postprocessor(modelSize: modelSize)
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
        // Clear previous cache
        cachedEmbedding = nil
        cachedEncoderOutputs = nil
        transformParams = nil
        
        // Preprocess image for SAM2 (returns CGImage)
        let (processedImage, transform) = try preprocessor.processForSAM2(image)
        self.transformParams = transform
        
        // Convert CGImage to MLFeatureValue
        let imageFeature = try preprocessor.createImageFeature(processedImage)
        
        // Run image encoder with performance monitoring
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "image": imageFeature
        ])
        
        #if DEBUG
        let encoderStartTime = CFAbsoluteTimeGetCurrent()
        #endif
        
        let encoderOutput = try imageEncoder.prediction(from: encoderInput)
        
        #if DEBUG
        let encoderTime = CFAbsoluteTimeGetCurrent() - encoderStartTime
        print("Image encoder inference time: \(String(format: "%.3f", encoderTime))s")
        #endif
        
        // Cache all encoder outputs - HuggingFace SAM2 needs multiple features
        var allEncoderOutputs: [String: MLMultiArray] = [:]
        
        // Extract all features from encoder
        for featureName in encoderOutput.featureNames {
            if let featureValue = encoderOutput.featureValue(for: featureName)?.multiArrayValue {
                allEncoderOutputs[featureName] = featureValue
            }
        }
        
        // Debug: print all encoder outputs
        print("Encoder outputs: \(allEncoderOutputs.keys.joined(separator: ", "))")
        
        // Find the main embedding
        let embedding: MLMultiArray?
        if let emb = allEncoderOutputs["image_embedding"] {
            embedding = emb
        } else if let emb = allEncoderOutputs["image_embeddings"] {
            embedding = emb
        } else {
            // Use the first available embedding
            embedding = allEncoderOutputs.values.first
        }
        
        guard let finalEmbedding = embedding else {
            throw SamError.invalidModelOutput("Could not extract image embeddings")
        }
        
        self.cachedEmbedding = finalEmbedding
        
        // Store all encoder outputs for later use
        self.cachedEncoderOutputs = allEncoderOutputs
    }
    
    /// Run mask prediction with prompts
    public func predict(
        points: [SamPoint] = [],
        box: SamBox? = nil,
        maskInput: SamMaskRef? = nil,
        options: SamOptions = SamOptions()
    ) async throws -> SamResult {
        
        guard let embedding = cachedEmbedding,
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
        
        // Prepare mask decoder inputs
        var decoderInputs: [String: Any] = [:]
        
        // Add all encoder outputs (includes feats_s0, feats_s1, image_embedding)
        for (key, value) in encoderOutputs {
            decoderInputs[key] = value
        }
        
        // Also provide alternative naming conventions
        decoderInputs["image_embedding"] = embedding
        decoderInputs["image_embeddings"] = embedding
        
        // Add prompt embeddings
        if let sparseEmbeddings = promptEmbeddings["sparse_embeddings"] {
            decoderInputs["sparse_embeddings"] = sparseEmbeddings
            decoderInputs["sparse_embedding"] = sparseEmbeddings  // Also try singular
        }
        if let denseEmbeddings = promptEmbeddings["dense_embeddings"] {
            decoderInputs["dense_embeddings"] = denseEmbeddings
            decoderInputs["dense_embedding"] = denseEmbeddings  // Also try singular
        }
        
        // Debug: print decoder inputs
        print("Decoder inputs: \(decoderInputs.keys.joined(separator: ", "))")
        
        // Run mask decoder with performance optimization
        let decoderInput = try MLDictionaryFeatureProvider(dictionary: decoderInputs)
        
        // Add performance timing in debug mode
        #if DEBUG
        let startTime = CFAbsoluteTimeGetCurrent()
        #endif
        
        let decoderOutput = try await withCheckedThrowingContinuation { continuation in
            Task.detached(priority: .high) {
                do {
                    let output = try self.maskDecoder.prediction(from: decoderInput)
                    continuation.resume(returning: output)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        
        #if DEBUG
        let decoderTime = CFAbsoluteTimeGetCurrent() - startTime
        print("Mask decoder inference time: \(String(format: "%.3f", decoderTime))s")
        #endif
        
        // Extract masks and scores - try different possible output keys
        let maskLogits: MLMultiArray?
        let iouPredictions: MLMultiArray?
        
        // Try common output keys for masks
        if let masks = decoderOutput.featureValue(for: "masks")?.multiArrayValue {
            maskLogits = masks
        } else if let masks = decoderOutput.featureValue(for: "low_res_masks")?.multiArrayValue {
            maskLogits = masks
        } else if let masks = decoderOutput.featureValue(for: "output_0")?.multiArrayValue {
            maskLogits = masks
        } else if let masks = decoderOutput.featureValue(for: "var_7056")?.multiArrayValue {
            // Some HuggingFace models use auto-generated names
            maskLogits = masks
        } else {
            maskLogits = nil
        }
        
        // Try common output keys for scores
        if let scores = decoderOutput.featureValue(for: "iou_predictions")?.multiArrayValue {
            iouPredictions = scores
        } else if let scores = decoderOutput.featureValue(for: "scores")?.multiArrayValue {
            iouPredictions = scores
        } else if let scores = decoderOutput.featureValue(for: "output_1")?.multiArrayValue {
            iouPredictions = scores
        } else if let scores = decoderOutput.featureValue(for: "var_7057")?.multiArrayValue {
            // Some HuggingFace models use auto-generated names
            iouPredictions = scores
        } else {
            iouPredictions = nil
        }
        
        guard let finalMaskLogits = maskLogits,
              let finalIouPredictions = iouPredictions else {
            let keys = decoderOutput.featureNames.joined(separator: ", ")
            throw SamError.invalidModelOutput("Missing masks or iou_predictions from decoder. Available keys: \(keys)")
        }
        
        // Debug: print mask decoder output shapes
        print("Mask decoder output shapes:")
        print("  Mask logits: \(finalMaskLogits.shape)")
        print("  IOU predictions: \(finalIouPredictions.shape)")
        
        // Postprocess results
        let result = try postprocessor.process(
            maskLogits: finalMaskLogits,
            iouPredictions: finalIouPredictions,
            transform: transform,
            options: options
        )
        
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
            // Provide empty box with proper shape
            let emptyBox = try MLMultiArray(shape: [1, 1, 4], dataType: .float32)
            // Set to full image bounds
            emptyBox[[0, 0, 0] as [NSNumber]] = NSNumber(value: 0.0)
            emptyBox[[0, 0, 1] as [NSNumber]] = NSNumber(value: 0.0)
            emptyBox[[0, 0, 2] as [NSNumber]] = NSNumber(value: 1024.0)
            emptyBox[[0, 0, 3] as [NSNumber]] = NSNumber(value: 1024.0)
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
    case small
    case base
    case large
    
    public var modelPrefix: String {
        switch self {
        case .tiny: return "SAM2Tiny"
        case .small: return "SAM2Small"
        case .base: return "SAM2Base"
        case .large: return "SAM2Large"
        }
    }
    
    public var inputSize: Int {
        return 1024  // All SAM2 models use 1024x1024 input
    }
    
    public var displayName: String {
        switch self {
        case .tiny: return "SAM 2 Tiny"
        case .small: return "SAM 2 Small"
        case .base: return "SAM 2 Base"
        case .large: return "SAM 2 Large"
        }
    }
}