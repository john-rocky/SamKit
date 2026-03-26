import Foundation
import CoreML
import Accelerate

/// Swift implementation of SAM prompt encoder for MobileSAM.
///
/// The CoreML decoder model does not include prompt encoding (due to
/// `index_put_` incompatibility), so encoding is performed in Swift
/// using weights exported from the PyTorch model.
public final class PromptEncoder {

    // MARK: - Properties

    /// Maximum number of user points supported by the CoreML decoder model.
    /// The model accepts sparse_embeddings with up to (maxPoints + 1) tokens
    /// (user points + 1 padding token).
    public static let maxPoints = 9

    private let embedDim: Int
    private let imageEmbeddingSize: (h: Int, w: Int)
    private let inputImageSize: (h: Int, w: Int)

    // Learned weights
    private let gaussianMatrix: [Float]  // [2, numPosFeats]
    private let numPosFeats: Int
    private let pointEmbeddings: [[Float]]  // 4 x [embedDim]
    private let notAPointEmbed: [Float]     // [embedDim]
    private let noMaskEmbed: [Float]        // [embedDim]

    // Cached dense embedding (always the same when no mask input)
    private let cachedDenseEmbedding: MLMultiArray

    // MARK: - Initialization

    /// Load prompt encoder weights from a JSON file.
    public init(weightsURL: URL) throws {
        let data = try Data(contentsOf: weightsURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        self.embedDim = json["embed_dim"] as! Int
        let ies = json["image_embedding_size"] as! [Int]
        self.imageEmbeddingSize = (ies[0], ies[1])
        let iis = json["input_image_size"] as! [Int]
        self.inputImageSize = (iis[0], iis[1])

        let gm = json["gaussian_matrix"] as! [[Double]]
        self.numPosFeats = gm[0].count  // 128
        self.gaussianMatrix = gm.flatMap { $0.map { Float($0) } }  // [2 * 128]

        let pe = json["point_embeddings"] as! [[Double]]
        self.pointEmbeddings = pe.map { $0.map { Float($0) } }

        self.notAPointEmbed = (json["not_a_point_embed"] as! [Double]).map { Float($0) }
        self.noMaskEmbed = (json["no_mask_embed"] as! [Double]).map { Float($0) }

        // Pre-build cached dense embedding [1, embedDim, H, W]
        let h = ies[0]; let w = ies[1]; let ed = self.embedDim
        let dense = try MLMultiArray(shape: [1, ed as NSNumber, h as NSNumber, w as NSNumber], dataType: .float32)
        let dPtr = dense.dataPointer.bindMemory(to: Float32.self, capacity: dense.count)
        let spatialSize = vDSP_Length(h * w)
        for c in 0..<ed {
            var val = self.noMaskEmbed[c]
            vDSP_vfill(&val, dPtr + c * Int(spatialSize), 1, spatialSize)
        }
        self.cachedDenseEmbedding = dense
    }

    // MARK: - Public API

    /// Encode point prompts into sparse embeddings and produce dense (no-mask) embeddings.
    /// - Returns: (sparse_embeddings [1, N, embedDim], dense_embeddings [1, embedDim, H, W])
    public func encode(
        points: [SamPoint],
        transform: TransformParams
    ) throws -> (MLMultiArray, MLMultiArray) {

        // Transform points to model coordinates and add padding token
        // Limit to maxPoints to stay within CoreML enumerated shapes
        let clampedPoints = points.count > Self.maxPoints
            ? Array(points.suffix(Self.maxPoints))
            : points

        var coords: [(Float, Float)] = []
        var labels: [Float] = []

        if clampedPoints.isEmpty {
            coords.append((Float(inputImageSize.w / 2), Float(inputImageSize.h / 2)))
            labels.append(-1)
        } else {
            for p in clampedPoints {
                let mp = transform.toModel(p)
                coords.append((Float(mp.x), Float(mp.y)))
                labels.append(Float(p.label.rawValue))
            }
        }

        // Add padding point (SAM always pads with one extra point)
        coords.append((0, 0))
        labels.append(-1)

        let numTokens = coords.count

        // Positional encoding for all points
        var sparse = [Float](repeating: 0, count: numTokens * embedDim)

        for i in 0..<numTokens {
            // Shift by 0.5 (center of pixel) as in Python
            let cx = coords[i].0 + 0.5
            let cy = coords[i].1 + 0.5

            // Position encoding via Random Fourier Features
            let pe = positionalEncoding(x: cx, y: cy)

            let offset = i * embedDim
            for d in 0..<embedDim {
                sparse[offset + d] = pe[d]
            }

            // Add label-specific learned embedding
            let label = labels[i]
            if label < 0 {
                // Not-a-point: zero out positional encoding, add not_a_point_embed
                for d in 0..<embedDim {
                    sparse[offset + d] = notAPointEmbed[d]
                }
            } else if label == 0 {
                // Negative point
                for d in 0..<embedDim {
                    sparse[offset + d] += pointEmbeddings[0][d]
                }
            } else {
                // Positive point
                for d in 0..<embedDim {
                    sparse[offset + d] += pointEmbeddings[1][d]
                }
            }
        }

        // Create MLMultiArray [1, numTokens, embedDim]
        let sparseArray = try MLMultiArray(
            shape: [1, numTokens as NSNumber, embedDim as NSNumber],
            dataType: .float32
        )
        let sparsePtr = sparseArray.dataPointer.bindMemory(to: Float32.self, capacity: sparse.count)
        sparse.withUnsafeBufferPointer { buf in
            sparsePtr.update(from: buf.baseAddress!, count: sparse.count)
        }

        return (sparseArray, cachedDenseEmbedding)
    }

    // MARK: - Private Methods

    /// Compute positional encoding for a single (x, y) coordinate.
    /// Matches Python: normalize to [0,1], matmul with gaussian_matrix, 2π, [sin, cos].
    private func positionalEncoding(x: Float, y: Float) -> [Float] {
        // Normalize to [0, 1]
        let nx = x / Float(inputImageSize.w)
        let ny = y / Float(inputImageSize.h)

        // Scale to [-1, 1] as in Python _pe_encoding
        let sx = 2.0 * nx - 1.0
        let sy = 2.0 * ny - 1.0

        // gaussianMatrix is [2, numPosFeats] stored row-major
        // result[j] = sx * gm[0][j] + sy * gm[1][j]
        var result = [Float](repeating: 0, count: embedDim)
        for j in 0..<numPosFeats {
            let val = sx * gaussianMatrix[j] + sy * gaussianMatrix[numPosFeats + j]
            let angle = val * 2.0 * .pi
            result[j] = sin(angle)
            result[numPosFeats + j] = cos(angle)
        }

        return result
    }

}
