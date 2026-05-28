import Foundation
import CoreML
import CoreGraphics
import CoreVideo
import Accelerate
import VideoToolbox

/// FastSAM — promptable "segment everything" on a single YOLOv8-seg forward pass.
///
/// Unlike `SamSession` / `Sam2Session` (image-encoder + prompt-decoder), FastSAM is a
/// YOLOv8-seg instance segmenter: one forward pass yields every object's box + mask, and
/// point / box prompts are resolved afterwards by *selecting* among those instances.
///
/// The CoreML model (see `conversion_scripts/convert_fastsam.py`) outputs four tensors:
///   boxes       [1, 4, A]        cx, cy, w, h in input-pixel coordinates
///   scores      [1, 1, A]        single "object" class, sigmoid-calibrated
///   mask_coeffs [1, 32, A]       per-anchor mask coefficients
///   mask_protos [1, 32, P, P]    prototypes; instance mask = sigmoid(coeffs · protos)
///
/// Two input shapes are supported, auto-detected from the model spec:
///   • **ImageType** (`CVPixelBuffer`): CoreML resizes + normalises on ANE/GPU — fastest path.
///     The Swift side just letterboxes a pixel buffer; no per-pixel float loop.
///   • **TensorType** (`MLMultiArray`): legacy path; Swift builds a normalised float tensor.
///
/// Performance shape:
///   • `setImage(_ pb: CVPixelBuffer)` — camera path, zero CGImage round-trip.
///   • `setImage(_ image: CGImage)`    — photo path, back-compat.
///   • `segmentEverythingMask`         — one `sgemm` for *all* instances, composites at proto
///     resolution (P×P, e.g. 160), one upsample by the view. ~16-40× cheaper than the per-
///     instance-at-input-size approach.
public final class FastSamSession {

    // MARK: - Model reference

    public struct ModelRef {
        public let detectorURL: URL
        public init(detectorURL: URL) { self.detectorURL = detectorURL }

        /// Load a bundled FastSAM detector by resource name (`FastSAM_s`, `FastSAM_s_512`, …).
        public static func bundled(_ name: String = "FastSAM_s") throws -> ModelRef {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc")
                    ?? Bundle.main.url(forResource: name, withExtension: "mlpackage") else {
                throw SamError.modelNotFound
            }
            return ModelRef(detectorURL: url)
        }
    }

    // MARK: - Options

    public struct Options {
        public let confidenceThreshold: Float
        public let iouThreshold: Float
        public let maxInstances: Int
        public let maskThreshold: Float

        public init(confidenceThreshold: Float = 0.4,
                    iouThreshold: Float = 0.9,
                    maxInstances: Int = 100,
                    maskThreshold: Float = 0.5) {
            self.confidenceThreshold = confidenceThreshold
            self.iouThreshold = iouThreshold
            self.maxInstances = maxInstances
            self.maskThreshold = maskThreshold
        }
    }

    public struct Instance {
        public let mask: SamMask
        public let box: SamBox
        public let score: Float
    }

    // MARK: - State

    private let detector: MLModel
    public let inputSize: Int
    public let protoSize: Int
    private let inputIsImage: Bool
    private let numMaskCoeffs = 32

    private var transform: LetterboxTransform?
    private var boxes: [Float] = []      // [4 * A]   (xywh in inputSize px)
    private var scores: [Float] = []     // [A]
    private var coeffs: [Float] = []     // [32 * A]
    private var protos: [Float] = []     // [32 * P * P]
    private var numAnchors = 0
    private var bufferPool: CVPixelBufferPool?   // reused letterbox buffers (camera path)

    /// Assign stable, persistent colours to objects across frames (IoU tracking) so the
    /// "everything" overlay stops flickering. Enable for real-time/video; leave off for
    /// one-shot photos (independent frames). Costs O(N·M) per frame — negligible for N,M<50.
    public var trackColors = false
    private struct Track { var id: Int; var box: SamBox; var color: (UInt8, UInt8, UInt8); var missed: Int }
    private var tracks: [Track] = []
    private var nextTrackID = 0
    private let trackIoUThreshold: Float = 0.3
    private let trackMaxAge = 8                  // keep a lost track this many frames (dropout hysteresis)

    /// Set true to print a per-frame timing breakdown to the console.
    public static var profile = false
    public private(set) var lastLetterboxMs = 0.0
    public private(set) var lastPredictMs = 0.0
    public private(set) var lastReadMs = 0.0
    public private(set) var lastPostMs = 0.0

    // MARK: - Init

    public init(model: ModelRef, config: RuntimeConfig = .bestAvailable) throws {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits.mlComputeUnits
        mlConfig.allowLowPrecisionAccumulationOnGPU = true
        let loaded = try MLModel(contentsOf: model.detectorURL, configuration: mlConfig)
        self.detector = loaded

        let desc = loaded.modelDescription.inputDescriptionsByName["image"]
        if let img = desc?.imageConstraint {
            self.inputSize = img.pixelsWide
            self.inputIsImage = true
        } else if let arr = desc?.multiArrayConstraint {
            let s = arr.shape.map { $0.intValue }
            self.inputSize = s.last ?? 640
            self.inputIsImage = false
        } else {
            throw SamError.invalidModelOutput("FastSAM: 'image' input not found in model")
        }
        // YOLOv8-seg always emits prototypes at 1/4 of the input.
        self.protoSize = inputSize / 4
        print("[FastSAM] init: inputSize=\(inputSize) protoSize=\(protoSize) imageInput=\(inputIsImage)")
    }

    public convenience init(modelName: String = "FastSAM_s",
                            config: RuntimeConfig = .bestAvailable) throws {
        try self.init(model: ModelRef.bundled(modelName), config: config)
    }

    // MARK: - Public API: setImage

    /// Photo / back-compat entry. Routes to the right input path automatically.
    public func setImage(_ image: CGImage) throws {
        if inputIsImage {
            let (pb, tf) = try letterboxCGImageToPixelBuffer(image)
            try runDetector(pixelBuffer: pb, transform: tf)
        } else {
            let (arr, tf) = try letterboxToMultiArray(image)
            try runDetector(multiArray: arr, transform: tf)
        }
    }

    /// Camera entry — feeds CoreML directly with the resized pixel buffer (no CGImage hop).
    public func setImage(_ pixelBuffer: CVPixelBuffer) throws {
        if inputIsImage {
            let t0 = CFAbsoluteTimeGetCurrent()
            let (pb, tf) = try letterboxPixelBuffer(pixelBuffer)
            lastLetterboxMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            try runDetector(pixelBuffer: pb, transform: tf)
            if Self.profile {
                print(String(format: "[FastSAM] letterbox=%.1f predict=%.1f read=%.1f ms",
                             lastLetterboxMs, lastPredictMs, lastReadMs))
            }
        } else {
            // Fallback for legacy TensorType models: pixel buffer → CGImage → MLMultiArray.
            var cg: CGImage?
            VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cg)
            guard let cgImage = cg else {
                throw SamError.preprocessingFailed("CVPixelBuffer → CGImage failed")
            }
            try setImage(cgImage)
        }
    }

    // MARK: - Public API: segmentation

    /// Segment everything; returns a list of instances (each with its own colour-tinted mask).
    /// For overlay / real-time use prefer `segmentEverythingMask`.
    public func segmentEverything(options: Options = Options()) throws -> [Instance] {
        let kept = try candidates(options: options)
        guard let tf = transform else { throw SamError.imageNotSet }
        guard !kept.isEmpty else { return [] }
        let plane = protoSize * protoSize
        let logits = try assembleBatchedLogits(kept: kept)
        let (w, h) = scaledProtoRegion(tf)
        var result: [Instance] = []
        result.reserveCapacity(kept.count)
        for (i, cand) in kept.enumerated() {
            let alpha = makeAlpha(forInstance: i,
                                  logits: logits, plane: plane,
                                  cand: cand, transform: tf,
                                  threshold: options.maskThreshold)
            let color = Self.paletteColor(i)
            let cg = Self.makeMaskCGImage(alpha: alpha, width: w, height: h, color: color)
            let mask = SamMask(width: w, height: h, logits: nil, alpha: alpha,
                               score: cand.score, cgImage: cg)
            result.append(Instance(mask: mask, box: cand.imageBox, score: cand.score))
        }
        return result
    }

    /// One composited overlay image at *proto* resolution (P-scaled, image-region only — same
    /// aspect as the original, so SwiftUI / Core Graphics scales it to display 1:1).
    public func segmentEverythingMask(options: Options = Options()) throws -> CGImage? {
        let tStart = CFAbsoluteTimeGetCurrent()
        defer {
            lastPostMs = (CFAbsoluteTimeGetCurrent() - tStart) * 1000
            if Self.profile { print(String(format: "[FastSAM] postproc=%.1f ms", lastPostMs)) }
        }
        let kept = try candidates(options: options)
        guard let tf = transform, !kept.isEmpty else { return nil }

        let N = kept.count
        let plane = protoSize * protoSize
        let tSgemm = CFAbsoluteTimeGetCurrent()
        let logits = try assembleBatchedLogits(kept: kept)
        if Self.profile {
            print(String(format: "[FastSAM]   N=%d candidates+sgemm=%.1f ms",
                         N, (CFAbsoluteTimeGetCurrent() - tSgemm) * 1000))
        }
        let (w, h) = scaledProtoRegion(tf)

        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let thr = options.maskThreshold
        let (padXp, padYp) = (padInProtoSpace(tf).x, padInProtoSpace(tf).y)
        let P = protoSize
        let scaleBox = Float(P) / Float(inputSize)

        // Stable colours via IoU tracking (video), else palette-by-rank (one-shot).
        let colors: [(UInt8, UInt8, UInt8)] = trackColors
            ? assignTrackColors(kept)
            : (0..<N).map { Self.paletteColor($0) }

        for n in 0..<N {
            let (cr, cg, cb) = colors[n]
            let bx0 = max(0, Int((kept[n].box640.x0 * scaleBox - Float(padXp)).rounded()))
            let by0 = max(0, Int((kept[n].box640.y0 * scaleBox - Float(padYp)).rounded()))
            let bx1 = min(w, Int((kept[n].box640.x1 * scaleBox - Float(padXp)).rounded()))
            let by1 = min(h, Int((kept[n].box640.y1 * scaleBox - Float(padYp)).rounded()))
            if bx1 <= bx0 || by1 <= by0 { continue }
            let logitBase = n * plane
            for y in by0..<by1 {
                let py = y + padYp
                let logitRow = logitBase + py * P
                let dstRow = y * w
                for x in bx0..<bx1 {
                    if logits[logitRow + (x + padXp)] > thr {
                        let dst = (dstRow + x) * 4
                        rgba[dst]     = cr
                        rgba[dst + 1] = cg
                        rgba[dst + 2] = cb
                        rgba[dst + 3] = 255
                    }
                }
            }
        }
        return Self.makeRGBACGImage(&rgba, width: w, height: h)
    }

    /// Resolve a tap (in original image coords) to the single best instance under it.
    public func segment(at point: CGPoint, options: Options = Options()) throws -> Instance? {
        let kept = try candidates(options: options)
        guard let tf = transform else { throw SamError.imageNotSet }
        let px = Float(point.x), py = Float(point.y)

        let containing = kept.filter { c in
            px >= c.imageBox.x0 && px <= c.imageBox.x1 &&
            py >= c.imageBox.y0 && py <= c.imageBox.y1
        }.sorted { area($0.imageBox) < area($1.imageBox) }
        guard !containing.isEmpty else { return nil }

        // Per-candidate single sgemv (cheap; usually 1-few containing the tap).
        for cand in containing {
            let alpha = singleInstanceAlpha(anchor: cand.anchor, cand: cand,
                                            transform: tf, threshold: options.maskThreshold)
            // Sample the mask at the tap location.
            let (w, h) = scaledProtoRegion(tf)
            let mx = Int(px / Float(tf.originalWidth) * Float(w))
            let my = Int(py / Float(tf.originalHeight) * Float(h))
            if mx >= 0, my >= 0, mx < w, my < h,
               alpha.withUnsafeBytes({ $0.bindMemory(to: UInt8.self)[my * w + mx] >= 128 }) {
                let cg = Self.makeMaskCGImage(alpha: alpha, width: w, height: h,
                                              color: Self.selectionColor)
                let mask = SamMask(width: w, height: h, logits: nil, alpha: alpha,
                                   score: cand.score, cgImage: cg)
                return Instance(mask: mask, box: cand.imageBox, score: cand.score)
            }
        }
        // Fallback: highest-score box that contained the point.
        if let cand = containing.max(by: { $0.score < $1.score }) {
            let alpha = singleInstanceAlpha(anchor: cand.anchor, cand: cand,
                                            transform: tf, threshold: options.maskThreshold)
            let (w, h) = scaledProtoRegion(tf)
            let cg = Self.makeMaskCGImage(alpha: alpha, width: w, height: h,
                                          color: Self.selectionColor)
            let mask = SamMask(width: w, height: h, logits: nil, alpha: alpha,
                               score: cand.score, cgImage: cg)
            return Instance(mask: mask, box: cand.imageBox, score: cand.score)
        }
        return nil
    }

    public func segment(in box: SamBox, options: Options = Options()) throws -> Instance? {
        let kept = try candidates(options: options)
        guard let tf = transform else { throw SamError.imageNotSet }
        guard let cand = kept.max(by: { iou($0.imageBox, box) < iou($1.imageBox, box) }),
              iou(cand.imageBox, box) > 0 else { return nil }
        let alpha = singleInstanceAlpha(anchor: cand.anchor, cand: cand,
                                        transform: tf, threshold: options.maskThreshold)
        let (w, h) = scaledProtoRegion(tf)
        let cg = Self.makeMaskCGImage(alpha: alpha, width: w, height: h, color: Self.selectionColor)
        let mask = SamMask(width: w, height: h, logits: nil, alpha: alpha,
                           score: cand.score, cgImage: cg)
        return Instance(mask: mask, box: cand.imageBox, score: cand.score)
    }

    public func clear() {
        transform = nil
        boxes.removeAll(keepingCapacity: false)
        scores.removeAll(keepingCapacity: false)
        coeffs.removeAll(keepingCapacity: false)
        protos.removeAll(keepingCapacity: false)
        numAnchors = 0
        resetTracking()
    }

    // MARK: - Detector run + output read

    private func runDetector(pixelBuffer pb: CVPixelBuffer, transform tf: LetterboxTransform) throws {
        self.transform = tf
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pb)
        ])
        let t0 = CFAbsoluteTimeGetCurrent()
        let out = try detector.prediction(from: input)
        let t1 = CFAbsoluteTimeGetCurrent()
        try readOutputs(out)
        lastPredictMs = (t1 - t0) * 1000
        lastReadMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000
    }

    private func runDetector(multiArray arr: MLMultiArray, transform tf: LetterboxTransform) throws {
        self.transform = tf
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": arr])
        let out = try detector.prediction(from: input)
        try readOutputs(out)
    }

    private func readOutputs(_ out: MLFeatureProvider) throws {
        guard let boxesArr = out.featureValue(for: "boxes")?.multiArrayValue,
              let scoresArr = out.featureValue(for: "scores")?.multiArrayValue,
              let coeffsArr = out.featureValue(for: "mask_coeffs")?.multiArrayValue,
              let protosArr = out.featureValue(for: "mask_protos")?.multiArrayValue else {
            throw SamError.invalidModelOutput("FastSAM detector missing one of boxes/scores/mask_coeffs/mask_protos")
        }
        numAnchors = boxesArr.shape.last?.intValue ?? 0
        boxes = readFloats(boxesArr)
        coeffs = readFloats(coeffsArr)
        protos = readFloats(protosArr)
        let sShape = scoresArr.shape.map { $0.intValue }
        let nc = sShape.count >= 2 ? sShape[1] : 1
        let raw = readFloats(scoresArr)
        scores = [Float](repeating: 0, count: numAnchors)
        for a in 0..<numAnchors {
            var best: Float = 0
            for c in 0..<nc { best = max(best, raw[c * numAnchors + a]) }
            scores[a] = best
        }
    }

    // MARK: - Candidates (decode + NMS)

    private struct Candidate {
        let anchor: Int
        let box640: SamBox
        let imageBox: SamBox
        let score: Float
    }

    private func candidates(options: Options) throws -> [Candidate] {
        guard let tf = transform, numAnchors > 0 else { throw SamError.imageNotSet }
        var cands: [Candidate] = []
        cands.reserveCapacity(64)
        for a in 0..<numAnchors {
            let s = scores[a]
            if s < options.confidenceThreshold { continue }
            let cx = boxes[0 * numAnchors + a]
            let cy = boxes[1 * numAnchors + a]
            let w  = boxes[2 * numAnchors + a]
            let h  = boxes[3 * numAnchors + a]
            let box640 = SamBox(x0: cx - w / 2, y0: cy - h / 2,
                                x1: cx + w / 2, y1: cy + h / 2)
            let img = tf.toImage(box640)
            let clamped = SamBox(
                x0: max(0, min(img.x0, Float(tf.originalWidth))),
                y0: max(0, min(img.y0, Float(tf.originalHeight))),
                x1: max(0, min(img.x1, Float(tf.originalWidth))),
                y1: max(0, min(img.y1, Float(tf.originalHeight)))
            )
            cands.append(Candidate(anchor: a, box640: box640, imageBox: clamped, score: s))
        }
        let kept = nms(cands, iouThreshold: options.iouThreshold)
        return Array(kept.prefix(options.maxInstances))
    }

    private func nms(_ cands: [Candidate], iouThreshold: Float) -> [Candidate] {
        guard !cands.isEmpty else { return [] }
        let order = cands.indices.sorted { cands[$0].score > cands[$1].score }
        var suppressed = Set<Int>()
        var kept: [Candidate] = []
        for i in order {
            if suppressed.contains(i) { continue }
            kept.append(cands[i])
            for j in order where j != i && !suppressed.contains(j) {
                if iou(cands[i].imageBox, cands[j].imageBox) > iouThreshold {
                    suppressed.insert(j)
                }
            }
        }
        return kept
    }

    // MARK: - IoU tracking (stable per-object colours across frames)

    /// Clear tracker state (call when starting a new, unrelated stream/photo).
    public func resetTracking() {
        tracks.removeAll(keepingCapacity: true)
        nextTrackID = 0
    }

    /// Match this frame's instances to existing tracks by box IoU (global-greedy: strongest
    /// overlaps first), carry over each track's colour, mint new colours for new objects, and
    /// age out tracks that briefly disappear. Returns one colour per kept instance.
    private func assignTrackColors(_ kept: [Candidate]) -> [(UInt8, UInt8, UInt8)] {
        let N = kept.count
        var result = [(UInt8, UInt8, UInt8)](repeating: Self.selectionColor, count: N)
        var candTrack = [Int](repeating: -1, count: N)     // matched old-track index per candidate
        var trackUsed = [Bool](repeating: false, count: tracks.count)
        var candAssigned = [Bool](repeating: false, count: N)

        // All (iou, cand, track) above threshold, strongest first.
        var pairs: [(Float, Int, Int)] = []
        for c in 0..<N {
            for t in 0..<tracks.count {
                let i = iou(kept[c].imageBox, tracks[t].box)
                if i > trackIoUThreshold { pairs.append((i, c, t)) }
            }
        }
        pairs.sort { $0.0 > $1.0 }
        for (_, c, t) in pairs {
            if candAssigned[c] || trackUsed[t] { continue }
            candAssigned[c] = true; trackUsed[t] = true; candTrack[c] = t
        }

        var newTracks: [Track] = []
        newTracks.reserveCapacity(N + tracks.count)
        for c in 0..<N {
            if candTrack[c] >= 0 {
                var tr = tracks[candTrack[c]]
                tr.box = kept[c].imageBox
                tr.missed = 0
                result[c] = tr.color
                newTracks.append(tr)
            } else {
                let color = Self.paletteColor(nextTrackID)
                newTracks.append(Track(id: nextTrackID, box: kept[c].imageBox, color: color, missed: 0))
                result[c] = color
                nextTrackID += 1
            }
        }
        // Age unmatched tracks so a 1-2 frame dropout keeps the same colour on return.
        for t in 0..<tracks.count where !trackUsed[t] {
            var tr = tracks[t]; tr.missed += 1
            if tr.missed <= trackMaxAge { newTracks.append(tr) }
        }
        tracks = newTracks
        return result
    }

    // MARK: - Mask assembly (batched + low-res)

    /// One sgemm for all kept instances: `logits[N, plane] = (coeffs[:,kept])^T · protos`.
    private func assembleBatchedLogits(kept: [Candidate]) throws -> [Float] {
        let N = kept.count
        let plane = protoSize * protoSize
        let nm = numMaskCoeffs

        // Gather coefficient matrix C[nm, N] row-major.
        var C = [Float](repeating: 0, count: nm * N)
        for n in 0..<N {
            let a = kept[n].anchor
            for k in 0..<nm { C[k * N + n] = coeffs[k * numAnchors + a] }
        }

        var logits = [Float](repeating: 0, count: N * plane)
        protos.withUnsafeBufferPointer { pb in
            C.withUnsafeBufferPointer { cb in
                // A = C^T (N × nm), B = protos (nm × plane), out (N × plane).
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            Int32(N), Int32(plane), Int32(nm),
                            1.0,
                            cb.baseAddress!, Int32(N),     // C: lda = cols of (nm × N) = N
                            pb.baseAddress!, Int32(plane), // protos: ldb = plane
                            0.0,
                            &logits, Int32(plane))         // logits[N, plane], ldc = plane
            }
        }
        sigmoid(&logits)
        return logits
    }

    /// Binary alpha for instance n (within `logits[N, plane]`), cropped to its box and sliced
    /// to the un-padded scaled-image region in proto space.
    private func makeAlpha(forInstance n: Int,
                           logits: [Float], plane: Int,
                           cand: Candidate, transform tf: LetterboxTransform,
                           threshold: Float) -> Data {
        let (w, h) = scaledProtoRegion(tf)
        let (padXp, padYp) = (padInProtoSpace(tf).x, padInProtoSpace(tf).y)
        let P = protoSize
        let scaleBox = Float(P) / Float(inputSize)
        let bx0 = max(0, Int((cand.box640.x0 * scaleBox - Float(padXp)).rounded()))
        let by0 = max(0, Int((cand.box640.y0 * scaleBox - Float(padYp)).rounded()))
        let bx1 = min(w, Int((cand.box640.x1 * scaleBox - Float(padXp)).rounded()))
        let by1 = min(h, Int((cand.box640.y1 * scaleBox - Float(padYp)).rounded()))

        var alpha = Data(count: w * h)
        alpha.withUnsafeMutableBytes { raw in
            let a = raw.bindMemory(to: UInt8.self).baseAddress!
            let base = n * plane
            if bx1 > bx0, by1 > by0 {
                for y in by0..<by1 {
                    let py = y + padYp
                    let logitRow = base + py * P
                    let dstRow = y * w
                    for x in bx0..<bx1 {
                        if logits[logitRow + (x + padXp)] > threshold {
                            a[dstRow + x] = 255
                        }
                    }
                }
            }
        }
        return alpha
    }

    /// Single-instance alpha for tap / box prompt (sgemv, no batched matmul needed).
    private func singleInstanceAlpha(anchor a: Int, cand: Candidate,
                                     transform tf: LetterboxTransform, threshold: Float) -> Data {
        let plane = protoSize * protoSize
        let nm = numMaskCoeffs
        var c = [Float](repeating: 0, count: nm)
        for k in 0..<nm { c[k] = coeffs[k * numAnchors + a] }
        var logit = [Float](repeating: 0, count: plane)
        protos.withUnsafeBufferPointer { pb in
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        Int32(nm), Int32(plane),
                        1.0, pb.baseAddress!, Int32(plane),
                        c, 1, 0.0, &logit, 1)
        }
        sigmoid(&logit)
        // Use the batched-style composite over the single logit by pretending N=1.
        return makeAlpha(forInstance: 0, logits: logit, plane: plane,
                         cand: cand, transform: tf, threshold: threshold)
    }

    // MARK: - Geometry helpers

    private struct LetterboxTransform {
        let scale: Float
        let padX: Int
        let padY: Int
        let scaledWidth: Int
        let scaledHeight: Int
        let originalWidth: Int
        let originalHeight: Int

        /// Map a box from input-pixel space back to original-image coordinates.
        func toImage(_ b: SamBox) -> SamBox {
            SamBox(x0: (b.x0 - Float(padX)) / scale, y0: (b.y0 - Float(padY)) / scale,
                   x1: (b.x1 - Float(padX)) / scale, y1: (b.y1 - Float(padY)) / scale)
        }
    }

    private func padInProtoSpace(_ tf: LetterboxTransform) -> (x: Int, y: Int) {
        let s = Float(protoSize) / Float(inputSize)
        return (Int((Float(tf.padX) * s).rounded()), Int((Float(tf.padY) * s).rounded()))
    }
    private func scaledProtoRegion(_ tf: LetterboxTransform) -> (Int, Int) {
        let s = Float(protoSize) / Float(inputSize)
        let w = max(1, Int((Float(tf.scaledWidth) * s).rounded()))
        let h = max(1, Int((Float(tf.scaledHeight) * s).rounded()))
        return (w, h)
    }

    // MARK: - Letterboxing

    private func letterboxToMultiArray(_ image: CGImage) throws -> (MLMultiArray, LetterboxTransform) {
        let S = inputSize
        let ow = image.width, oh = image.height
        let scale = Float(S) / Float(max(ow, oh))
        let sw = Int(Float(ow) * scale), sh = Int(Float(oh) * scale)
        let padX = (S - sw) / 2, padY = (S - sh) / 2

        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: S, height: S, bitsPerComponent: 8,
                                  bytesPerRow: S * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw SamError.preprocessingFailed("Failed to create context")
        }
        ctx.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: S, height: S))
        ctx.draw(image, in: CGRect(x: padX, y: padY, width: sw, height: sh))
        guard let data = ctx.data else { throw SamError.preprocessingFailed("No pixel data") }
        let px = data.bindMemory(to: UInt8.self, capacity: S * S * 4)

        let arr = try MLMultiArray(shape: [1, 3, S as NSNumber, S as NSNumber], dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
        let plane = S * S
        for y in 0..<S {
            for x in 0..<S {
                let s = (y * S + x) * 4
                let d = y * S + x
                ptr[d] = Float32(px[s]) / 255
                ptr[plane + d] = Float32(px[s + 1]) / 255
                ptr[2 * plane + d] = Float32(px[s + 2]) / 255
            }
        }
        let tf = LetterboxTransform(scale: scale, padX: padX, padY: padY,
                                    scaledWidth: sw, scaledHeight: sh,
                                    originalWidth: ow, originalHeight: oh)
        return (arr, tf)
    }

    /// Letterbox a `CGImage` straight into a 32BGRA `CVPixelBuffer` for ImageType models. The
    /// Swift side does no per-pixel float work — CoreML handles `/255` + RGB conversion on
    /// the ANE/GPU.
    private func letterboxCGImageToPixelBuffer(_ image: CGImage) throws -> (CVPixelBuffer, LetterboxTransform) {
        let S = inputSize
        let ow = image.width, oh = image.height
        let scale = Float(S) / Float(max(ow, oh))
        let sw = Int(Float(ow) * scale), sh = Int(Float(oh) * scale)
        let padX = (S - sw) / 2, padY = (S - sh) / 2

        let pb = try makeBGRABuffer(width: S, height: S)
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let base = CVPixelBufferGetBaseAddress(pb)!
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        memset(base, 0x80, bpr * S)  // gray pad (BGRA = 0x80808080)

        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue:
            CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        if let ctx = CGContext(data: base, width: S, height: S, bitsPerComponent: 8,
                               bytesPerRow: bpr, space: cs, bitmapInfo: info.rawValue) {
            ctx.draw(image, in: CGRect(x: padX, y: padY, width: sw, height: sh))
        }
        let tf = LetterboxTransform(scale: scale, padX: padX, padY: padY,
                                    scaledWidth: sw, scaledHeight: sh,
                                    originalWidth: ow, originalHeight: oh)
        return (pb, tf)
    }

    /// Letterbox a source pixel buffer (e.g. camera frame) into the model's input pixel buffer
    /// using vImage — no CGImage allocation per frame.
    private func letterboxPixelBuffer(_ src: CVPixelBuffer) throws -> (CVPixelBuffer, LetterboxTransform) {
        let S = inputSize
        let ow = CVPixelBufferGetWidth(src), oh = CVPixelBufferGetHeight(src)
        let scale = Float(S) / Float(max(ow, oh))
        let sw = Int(Float(ow) * scale), sh = Int(Float(oh) * scale)
        let padX = (S - sw) / 2, padY = (S - sh) / 2

        let dst = try makeBGRABuffer(width: S, height: S)
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
            CVPixelBufferUnlockBaseAddress(dst, [])
        }
        let dstBase = CVPixelBufferGetBaseAddress(dst)!
        let dstBpr = CVPixelBufferGetBytesPerRow(dst)
        memset(dstBase, 0x80, dstBpr * S)

        var srcBuf = vImage_Buffer(
            data: CVPixelBufferGetBaseAddress(src),
            height: vImagePixelCount(oh),
            width:  vImagePixelCount(ow),
            rowBytes: CVPixelBufferGetBytesPerRow(src)
        )
        var subBuf = vImage_Buffer(
            data: dstBase.advanced(by: padY * dstBpr + padX * 4),
            height: vImagePixelCount(sh),
            width:  vImagePixelCount(sw),
            rowBytes: dstBpr
        )
        vImageScale_ARGB8888(&srcBuf, &subBuf, nil, vImage_Flags(kvImageHighQualityResampling))

        let tf = LetterboxTransform(scale: scale, padX: padX, padY: padY,
                                    scaledWidth: sw, scaledHeight: sh,
                                    originalWidth: ow, originalHeight: oh)
        return (dst, tf)
    }

    /// Pooled BGRA buffers so the camera path doesn't allocate a fresh IOSurface every frame.
    private func makeBGRABuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        if bufferPool == nil {
            let pbAttrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil,
                                    [kCVPixelBufferPoolMinimumBufferCountKey as String: 3] as CFDictionary,
                                    pbAttrs as CFDictionary, &pool)
            bufferPool = pool
        }
        if let pool = bufferPool {
            var pb: CVPixelBuffer?
            if CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) == kCVReturnSuccess, let buffer = pb {
                return buffer
            }
        }
        // Fallback: direct allocation.
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let status = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                                         attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buffer = pb else {
            throw SamError.preprocessingFailed("CVPixelBuffer create failed (status=\(status))")
        }
        return buffer
    }

    // MARK: - Numeric / dtype helpers

    private func sigmoid(_ v: inout [Float]) {
        var n = Int32(v.count)
        var lo: Float = -50, hi: Float = 50
        vDSP_vclip(v, 1, &lo, &hi, &v, 1, vDSP_Length(v.count))
        vDSP_vneg(v, 1, &v, 1, vDSP_Length(v.count))
        vvexpf(&v, v, &n)
        var one: Float = 1
        vDSP_vsadd(v, 1, &one, &v, 1, vDSP_Length(v.count))
        vvrecf(&v, v, &n)
    }

    private func area(_ b: SamBox) -> Float { max(0, b.x1 - b.x0) * max(0, b.y1 - b.y0) }

    private func iou(_ a: SamBox, _ b: SamBox) -> Float {
        let x0 = max(a.x0, b.x0), y0 = max(a.y0, b.y0)
        let x1 = min(a.x1, b.x1), y1 = min(a.y1, b.y1)
        let inter = max(0, x1 - x0) * max(0, y1 - y0)
        let union = area(a) + area(b) - inter
        return union > 0 ? inter / union : 0
    }

    /// Stride-aware read of an MLMultiArray. ANE outputs are usually row-padded for SIMD
    /// alignment, so a naive flat copy is wrong. Fast paths:
    ///   • fully contiguous: one `memcpy`.
    ///   • only outer dim(s) padded (typical): per-leaf `memcpy` of the inner contiguous block.
    ///   • fallback: general strided gather.
    private func readFloats(_ a: MLMultiArray) -> [Float] {
        let count = a.count
        var out = [Float](repeating: 0, count: count)
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        let dtype = a.dataType
        let raw = a.dataPointer

        // Element size in bytes. FP16 CoreML models emit **Float16** outputs — converting those
        // per-element via NSNumber (`a[i].floatValue`) costs ~180ms/frame for ~700K elements,
        // so all dtypes go through a bulk, contiguous-block conversion instead.
        let elemSize: Int
        switch dtype {
        case .float16: elemSize = 2
        case .double:  elemSize = 8
        default:       elemSize = 4   // float32 / int32
        }

        // Convert `n` contiguous source elements (at element offset `srcElem`) into out[dstOff...].
        func copyBlock(_ dst: UnsafeMutablePointer<Float>, _ srcElem: Int, _ dstOff: Int, _ n: Int) {
            let srcBytes = raw.advanced(by: srcElem * elemSize)
            switch dtype {
            case .float32:
                memcpy(dst.advanced(by: dstOff), srcBytes, n * 4)
            case .float16:
                var s = vImage_Buffer(data: srcBytes, height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
                var d = vImage_Buffer(data: dst.advanced(by: dstOff), height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
                vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
            case .double:
                let sp = srcBytes.assumingMemoryBound(to: Double.self)
                for i in 0..<n { dst[dstOff + i] = Float(sp[i]) }
            default: // int32
                let sp = srcBytes.assumingMemoryBound(to: Int32.self)
                for i in 0..<n { dst[dstOff + i] = Float(sp[i]) }
            }
        }

        // Find the deepest k such that strides[k..] form a C-contiguous block.
        var k = shape.count - 1
        var blockSize = 1
        while k >= 0 && strides[k] == blockSize {
            blockSize *= shape[k]
            k -= 1
        }
        out.withUnsafeMutableBufferPointer { dst in
            let dstPtr = dst.baseAddress!
            if k < 0 {
                copyBlock(dstPtr, 0, 0, count)   // fully contiguous: one bulk convert
                return
            }
            let outerDims = k + 1
            var coord = [Int](repeating: 0, count: outerDims)
            var dstOffset = 0
            while true {
                var srcElem = 0
                for d in 0..<outerDims { srcElem += coord[d] * strides[d] }
                copyBlock(dstPtr, srcElem, dstOffset, blockSize)
                dstOffset += blockSize
                var d = outerDims - 1
                var done = false
                while d >= 0 {
                    coord[d] += 1
                    if coord[d] < shape[d] { break }
                    coord[d] = 0
                    d -= 1
                    if d < 0 { done = true; break }
                }
                if done { break }
            }
        }
        return out
    }

    // MARK: - Colour

    static let selectionColor: (UInt8, UInt8, UInt8) = (30, 144, 255)  // dodger blue (matches SAM)

    static func paletteColor(_ i: Int) -> (UInt8, UInt8, UInt8) {
        let hue = (Float(i) * 0.61803398875).truncatingRemainder(dividingBy: 1)
        return hsv(hue, 0.75, 1.0)
    }

    private static func hsv(_ h: Float, _ s: Float, _ v: Float) -> (UInt8, UInt8, UInt8) {
        let i = Int(h * 6)
        let f = h * 6 - Float(i)
        let p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s)
        let (r, g, b): (Float, Float, Float)
        switch i % 6 {
        case 0: (r, g, b) = (v, t, p)
        case 1: (r, g, b) = (q, v, p)
        case 2: (r, g, b) = (p, v, t)
        case 3: (r, g, b) = (p, q, v)
        case 4: (r, g, b) = (t, p, v)
        default: (r, g, b) = (v, p, q)
        }
        return (UInt8(r * 255), UInt8(g * 255), UInt8(b * 255))
    }

    static func makeMaskCGImage(alpha: Data, width: Int, height: Int,
                                color: (UInt8, UInt8, UInt8)) -> CGImage {
        let count = width * height
        var pixels = [UInt8](repeating: 0, count: count * 4)
        alpha.withUnsafeBytes { raw in
            let a = raw.bindMemory(to: UInt8.self).baseAddress!
            let rf = Float(color.0), gf = Float(color.1), bf = Float(color.2)
            for i in 0..<count {
                let af = Float(a[i]) / 255
                let p = i * 4
                pixels[p]     = UInt8(rf * af)
                pixels[p + 1] = UInt8(gf * af)
                pixels[p + 2] = UInt8(bf * af)
                pixels[p + 3] = a[i]
            }
        }
        return makeRGBACGImage(&pixels, width: width, height: height)
    }

    static func makeRGBACGImage(_ pixels: inout [UInt8], width: Int, height: Int) -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let provider = CGDataProvider(data: NSData(bytes: &pixels, length: pixels.count))!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: cs, bitmapInfo: info, provider: provider,
                       decode: nil, shouldInterpolate: true, intent: .defaultIntent)!
    }
}
