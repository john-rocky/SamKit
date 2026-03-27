import Foundation
import CoreML
import CoreGraphics
import Accelerate
import SAMKit

/// Open-vocabulary object detection using YOLO-World + CLIP.
///
/// The detection pipeline splits text-visual similarity out of CoreML
/// and computes it in Swift to work around iOS CoreML limitations:
///
/// 1. Visual model (CoreML): image → boxes + visual features
/// 2. CLIP text encoder (CoreML): text → text embeddings
/// 3. Similarity (Swift/Accelerate): features × text → class scores
public final class TextDetector {

    // MARK: - Properties

    private let textEncoder: MLModel
    private let visualModel: MLModel
    private let tokenizer: CLIPTokenizer
    private let preprocessor: GroundingPreprocessor
    private let config: RuntimeConfig
    private let maxClasses: Int
    private let contextLength: Int

    /// Per-level logit_scale and bias from BNContrastiveHead
    private let cv4Levels: [(logitScale: Float, bias: Float)]
    private let anchorsPerLevel: [Int]

    /// Cache text embeddings by query string
    private var textEmbeddingCache: [String: [Float]] = [:]

    // MARK: - Initialization

    public init(model: GroundingModelRef, config: RuntimeConfig = .bestAvailable) throws {
        self.config = config
        self.maxClasses = model.maxClasses
        self.contextLength = model.contextLength

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits.mlComputeUnits
        mlConfig.allowLowPrecisionAccumulationOnGPU = true

        self.textEncoder = try MLModel(contentsOf: model.textEncoderURL, configuration: mlConfig)
        self.visualModel = try MLModel(contentsOf: model.detectorURL, configuration: mlConfig)
        self.tokenizer = try CLIPTokenizer(vocabularyURL: model.vocabularyURL)
        self.preprocessor = GroundingPreprocessor(inputSize: model.inputSize)

        // Load cv4 parameters
        let cv4Data = try Data(contentsOf: model.cv4ParamsURL)
        guard let cv4Json = try JSONSerialization.jsonObject(with: cv4Data) as? [String: Any],
              let levels = cv4Json["levels"] as? [[String: Any]],
              let anchors = cv4Json["anchors_per_level"] as? [Int] else {
            throw GroundingError.modelNotFound("Invalid cv4_params.json format")
        }

        self.cv4Levels = levels.map {
            (logitScale: ($0["logit_scale"] as? Double ?? 0).float,
             bias: ($0["bias"] as? Double ?? 0).float)
        }
        self.anchorsPerLevel = anchors
    }

    // MARK: - Text Encoding

    /// Encode text queries to L2-normalized CLIP embeddings (cached)
    public func encodeText(_ queries: [String]) throws -> [[Float]] {
        var results: [[Float]] = []
        for query in queries {
            if let cached = textEmbeddingCache[query] {
                results.append(cached)
                continue
            }

            // Tokenize: pad to maxClasses slots, single query in slot 0
            let tokenArray = try MLMultiArray(
                shape: [maxClasses as NSNumber, contextLength as NSNumber],
                dataType: .int32
            )
            let tokenPtr = tokenArray.dataPointer.bindMemory(to: Int32.self, capacity: maxClasses * contextLength)
            // Zero fill
            memset(tokenPtr, 0, maxClasses * contextLength * 4)
            // Fill slot 0 with query tokens
            let tokens = tokenizer.tokenize(query)
            for j in 0..<contextLength {
                tokenPtr[j] = Int32(tokens[j])
            }

            let input = try MLDictionaryFeatureProvider(dictionary: ["text_tokens": tokenArray])
            let output = try textEncoder.prediction(from: input)
            guard let embeddings = output.featureValue(for: "text_embeddings")?.multiArrayValue else {
                throw GroundingError.invalidModelOutput("Missing text_embeddings")
            }

            // Extract slot 0 embedding [512] and convert to Float32
            let dim = 512
            var embedding = [Float](repeating: 0, count: dim)
            let emb = readMLMultiArrayAsFloat(embeddings)
            for i in 0..<dim { embedding[i] = emb[i] }

            // L2 normalize (should already be normalized, but ensure)
            var norm: Float = 0
            vDSP_svesq(embedding, 1, &norm, vDSP_Length(dim))
            norm = sqrt(norm)
            if norm > 1e-8 {
                var scale = 1.0 / norm
                vDSP_vsmul(embedding, 1, &scale, &embedding, 1, vDSP_Length(dim))
            }

            textEmbeddingCache[query] = embedding
            results.append(embedding)
        }
        return results
    }

    // MARK: - Detection

    /// Detect objects matching text queries in the given image.
    public func detect(
        image: CGImage,
        queries: [String],
        options: TextPromptOptions = TextPromptOptions()
    ) throws -> [GroundingResult] {
        let t0 = CFAbsoluteTimeGetCurrent()

        // 1. Encode text
        let textEmbeddings = try encodeText(queries) // [[Float]] each [512]
        let t1 = CFAbsoluteTimeGetCurrent()

        // 2. Run visual model with txt_feats (C2fAttn uses it for text-guided attention)
        let (processedImage, transform) = try preprocessor.process(image)
        let txtFeatsArray = try buildTxtFeatsArray(textEmbeddings)
        let visualInput = try MLDictionaryFeatureProvider(dictionary: [
            "image": processedImage,
            "txt_feats": txtFeatsArray,
        ])
        let visualOutput = try visualModel.prediction(from: visualInput)

        guard let boxesArray = visualOutput.featureValue(for: "boxes")?.multiArrayValue,
              let featsArray = visualOutput.featureValue(for: "features")?.multiArrayValue else {
            throw GroundingError.invalidModelOutput("Missing boxes or features")
        }
        let t2 = CFAbsoluteTimeGetCurrent()

        // 3. Compute text-visual similarity in Swift
        let boxes = readMLMultiArrayAsFloat(boxesArray)     // [1, 4, 8400] flattened
        let features = readMLMultiArrayAsFloat(featsArray)   // [1, 512, 8400] flattened

        let featShape = featsArray.shape.map { $0.intValue }
        let numAnchors: Int
        let embedDim: Int
        if featShape.count == 3 {
            embedDim = featShape[1]; numAnchors = featShape[2]
        } else {
            embedDim = 512; numAnchors = 8400
        }

        // Compute scores for each query
        var allDetections: [DetectorBox] = []
        var allScores: [Float] = []
        var allClassIds: [Int] = []

        for (queryIdx, textEmb) in textEmbeddings.enumerated() {
            // For each anchor: score = dot(features[:, anchor], textEmb) * scale + bias
            for anchor in 0..<numAnchors {
                var dot: Float = 0
                // features layout: [512, 8400], element [c, n] = features[c * 8400 + n]
                for c in 0..<embedDim {
                    dot += features[c * numAnchors + anchor] * textEmb[c]
                }

                // Apply per-level logit_scale and bias
                let (logitScale, bias) = levelParams(for: anchor)
                let logit = dot * exp(logitScale) + bias
                let score = 1.0 / (1.0 + exp(-min(max(logit, -50), 50)))

                if score < options.confidenceThreshold { continue }

                // Read box: [4, 8400], layout [ch, anchor]
                let cx = boxes[0 * numAnchors + anchor]
                let cy = boxes[1 * numAnchors + anchor]
                let w  = boxes[2 * numAnchors + anchor]
                let h  = boxes[3 * numAnchors + anchor]

                let box = DetectorBox(
                    x0: cx - w / 2, y0: cy - h / 2,
                    x1: cx + w / 2, y1: cy + h / 2
                )
                let imgBox = preprocessor.toImageCoordinates(box, transform: transform)
                let clamped = DetectorBox(
                    x0: max(0, min(imgBox.x0, Float(transform.originalWidth))),
                    y0: max(0, min(imgBox.y0, Float(transform.originalHeight))),
                    x1: max(0, min(imgBox.x1, Float(transform.originalWidth))),
                    y1: max(0, min(imgBox.y1, Float(transform.originalHeight)))
                )

                allDetections.append(clamped)
                allScores.append(score)
                allClassIds.append(queryIdx)
            }
        }

        let t3 = CFAbsoluteTimeGetCurrent()

        // 4. NMS + filter
        let kept = preprocessor.nms(
            boxes: allDetections, scores: allScores, labels: allClassIds,
            iouThreshold: options.nmsThreshold
        )

        let topIndices = Array(kept.prefix(options.maxDetections))
        guard !topIndices.isEmpty else {
            print("[SAMKitGrounding] text=\(Int((t1-t0)*1000))ms visual=\(Int((t2-t1)*1000))ms match=\(Int((t3-t2)*1000))ms | no detections")
            return []
        }

        // Filter: keep only within 30% of top score
        let maxScore = topIndices.map { allScores[$0] }.max() ?? 0
        let cutoff = maxScore * 0.3

        let results: [GroundingResult] = topIndices.compactMap { idx in
            guard allScores[idx] >= cutoff else { return nil }
            let box = allDetections[idx]
            let label = allClassIds[idx] < queries.count ? queries[allClassIds[idx]] : "object"
            return GroundingResult(
                box: SamBox(x0: box.x0, y0: box.y0, x1: box.x1, y1: box.y1),
                confidence: allScores[idx],
                label: label
            )
        }

        let t4 = CFAbsoluteTimeGetCurrent()
        print("[SAMKitGrounding] text=\(Int((t1-t0)*1000))ms visual=\(Int((t2-t1)*1000))ms match=\(Int((t3-t2)*1000))ms post=\(Int((t4-t3)*1000))ms | \(results.count) detections")

        return results
    }

    /// Clear cached text embeddings
    public func clearCache() {
        textEmbeddingCache.removeAll()
        tokenizer.clearCache()
    }

    // MARK: - Helpers

    /// Build txt_feats MLMultiArray [1, maxClasses, 512] for the visual model's C2fAttn
    private func buildTxtFeatsArray(_ embeddings: [[Float]]) throws -> MLMultiArray {
        let dim = 512
        let array = try MLMultiArray(shape: [1, maxClasses as NSNumber, dim as NSNumber], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: maxClasses * dim)
        // Zero fill
        memset(ptr, 0, maxClasses * dim * 4)
        // Fill with query embeddings
        for (i, emb) in embeddings.prefix(maxClasses).enumerated() {
            for j in 0..<min(emb.count, dim) {
                ptr[i * dim + j] = emb[j]
            }
        }
        return array
    }

    /// Get logit_scale and bias for the given anchor index based on detection level
    private func levelParams(for anchor: Int) -> (logitScale: Float, bias: Float) {
        var offset = 0
        for (i, count) in anchorsPerLevel.enumerated() {
            if anchor < offset + count {
                return cv4Levels[i]
            }
            offset += count
        }
        return cv4Levels.last ?? (0, 0)
    }

    /// Read MLMultiArray data as Float32 array (handles FP16 conversion)
    private func readMLMultiArrayAsFloat(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)

        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
            for i in 0..<count { result[i] = ptr[i] }
        } else {
            // Float16
            let ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count { result[i] = float16ToFloat32(ptr[i]) }
        }

        return result
    }

    /// Convert a Float16 value (stored as UInt16) to Float32
    private func float16ToFloat32(_ h: UInt16) -> Float {
        let sign     = UInt32(h >> 15) & 0x1
        let exponent = UInt32(h >> 10) & 0x1F
        let mantissa = UInt32(h)       & 0x3FF

        var f: UInt32
        if exponent == 0 {
            if mantissa == 0 {
                f = sign << 31
            } else {
                var e = exponent; var m = mantissa
                while (m & 0x400) == 0 { m <<= 1; e &-= 1 }
                e += 1; m &= ~UInt32(0x400)
                f = (sign << 31) | ((e + 112) << 23) | (m << 13)
            }
        } else if exponent == 31 {
            f = (sign << 31) | 0x7F800000 | (mantissa << 13)
        } else {
            f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13)
        }

        return Float(bitPattern: f)
    }
}

private extension Double {
    var float: Float { Float(self) }
}
