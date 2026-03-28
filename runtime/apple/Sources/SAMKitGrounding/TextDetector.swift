import Foundation
import CoreML
import CoreGraphics
import Accelerate
import SAMKit

/// Open-vocabulary object detection using YOLO-World + CLIP.
///
/// 1. Visual model (CoreML): image + text embeddings → boxes + class scores
///    The model includes the full BNContrastiveHead scoring internally,
///    so scores are already sigmoid-calibrated confidence values.
/// 2. CLIP text encoder (CoreML): text → text embeddings
/// 3. Post-processing (Swift): NMS + filtering
public final class TextDetector {

    // MARK: - Properties

    private let textEncoder: MLModel
    private let visualModel: MLModel
    private let tokenizer: CLIPTokenizer
    private let preprocessor: GroundingPreprocessor
    private let config: RuntimeConfig
    private let maxClasses: Int
    private let contextLength: Int

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

            let tokenArray = try MLMultiArray(
                shape: [maxClasses as NSNumber, contextLength as NSNumber],
                dataType: .int32
            )
            let tokenPtr = tokenArray.dataPointer.bindMemory(to: Int32.self, capacity: maxClasses * contextLength)
            memset(tokenPtr, 0, maxClasses * contextLength * 4)
            let tokens = tokenizer.tokenize(query)
            for j in 0..<contextLength {
                tokenPtr[j] = Int32(tokens[j])
            }

            let input = try MLDictionaryFeatureProvider(dictionary: ["text_tokens": tokenArray])
            let output = try textEncoder.prediction(from: input)
            guard let embeddings = output.featureValue(for: "text_embeddings")?.multiArrayValue else {
                throw GroundingError.invalidModelOutput("Missing text_embeddings")
            }

            let dim = 512
            var embedding = [Float](repeating: 0, count: dim)
            let emb = readMLMultiArrayAsFloat(embeddings)
            for i in 0..<dim { embedding[i] = emb[i] }

            // L2 normalize
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
        let textEmbeddings = try encodeText(queries)
        let t1 = CFAbsoluteTimeGetCurrent()

        // 2. Run visual model — scores are computed inside the model (BNContrastiveHead)
        let (processedImage, transform) = try preprocessor.process(image)
        let txtFeatsArray = try buildTxtFeatsArray(textEmbeddings)
        let visualInput = try MLDictionaryFeatureProvider(dictionary: [
            "image": processedImage,
            "txt_feats": txtFeatsArray,
        ])
        let visualOutput = try visualModel.prediction(from: visualInput)

        guard let boxesArray = visualOutput.featureValue(for: "boxes")?.multiArrayValue,
              let scoresArray = visualOutput.featureValue(for: "scores")?.multiArrayValue else {
            throw GroundingError.invalidModelOutput("Missing boxes or scores")
        }
        let t2 = CFAbsoluteTimeGetCurrent()

        // 3. Read pre-computed scores and filter
        let boxes = readMLMultiArrayAsFloat(boxesArray)     // [1, 4, 8400]
        let scores = readMLMultiArrayAsFloat(scoresArray)   // [1, NC, 8400]

        let scoresShape = scoresArray.shape.map { $0.intValue }
        let numClasses = scoresShape.count >= 2 ? scoresShape[1] : maxClasses
        let numAnchors = scoresShape.count >= 3 ? scoresShape[2] : 8400

        var allDetections: [DetectorBox] = []
        var allScores: [Float] = []
        var allClassIds: [Int] = []

        for queryIdx in 0..<min(queries.count, numClasses) {
            let classOffset = queryIdx * numAnchors

            for anchor in 0..<numAnchors {
                let score = scores[classOffset + anchor]
                if score < options.confidenceThreshold { continue }

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
            print("[SAMKitGrounding] text=\(Int((t1-t0)*1000))ms visual=\(Int((t2-t1)*1000))ms post=\(Int((t3-t2)*1000))ms | no detections")
            return []
        }

        // Filter: keep only scores within 50% of top score
        let maxScore = topIndices.map { allScores[$0] }.max() ?? 0
        let cutoff = maxScore * 0.5

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
        print("[SAMKitGrounding] text=\(Int((t1-t0)*1000))ms visual=\(Int((t2-t1)*1000))ms post=\(Int((t3-t2)*1000))ms nms=\(Int((t4-t3)*1000))ms | \(results.count) detections (maxScore=\(String(format: "%.3f", maxScore)))")

        return results
    }

    /// Clear cached text embeddings
    public func clearCache() {
        textEmbeddingCache.removeAll()
        tokenizer.clearCache()
    }

    // MARK: - Helpers

    /// Build txt_feats MLMultiArray [1, maxClasses, 512] for the visual model
    private func buildTxtFeatsArray(_ embeddings: [[Float]]) throws -> MLMultiArray {
        let dim = 512
        let array = try MLMultiArray(shape: [1, maxClasses as NSNumber, dim as NSNumber], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: maxClasses * dim)
        memset(ptr, 0, maxClasses * dim * 4)
        for (i, emb) in embeddings.prefix(maxClasses).enumerated() {
            for j in 0..<min(emb.count, dim) {
                ptr[i * dim + j] = emb[j]
            }
        }
        return array
    }

    /// Read MLMultiArray data as Float32 array (handles FP16 conversion)
    private func readMLMultiArrayAsFloat(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)

        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
            for i in 0..<count { result[i] = ptr[i] }
        } else {
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
