
# SAMKit Strategy (iOS: Core ML / Android: TFLite)
*AI Agent Implementation Playbook — Core inference parity with Python SAM + SamView UI*

---

## 0. Objectives & Scope
- **Goal**: Implement **SAMKit**, a mobile library that reproduces the Python `SamPredictor` flow — *image embedding → prompt (points/box/prev mask) → mask* — on **iOS (Core ML)** and **Android (TensorFlow Lite)**.
- **Components**:  
  1) **Core Inference**: model pre/post-processing, encoder/decoder execution, coordinate transforms, embedding cache.  
  2) **SamView UI**: on-canvas point/box inputs, multi-mask preview, editing/combination, export.
- **Out of Scope**: training/fine-tuning, cloud inference (separate plans if needed).

---

## 1. Assumptions & Constraints
- **Platforms**
  - iOS: prefer **iOS 15+** (async/await, Metal optimizations, ANE/Core ML acceleration).
  - Android: **API 26+**; prefer TFLite GPU Delegate/NNAPI when available.
- **Model Strategy**
  - **iOS (Core ML)**: SAM ViT-B for quality or **MobileSAM**-like variants for speed.
  - **Android (TFLite)**: prioritize **MobileSAM-class**/TFLite-friendly models (full SAM decoders with heavy transformer ops can be TFLite-incompatible).
- **I/O Resolution**: long side **S = 1024** as default (square padding). Provide 512/768 profiles for mid devices.
- **Licensing**: verify model and converter licenses (SAM/derivatives follow their respective terms).

---

## 2. High-Level Architecture
```
SAMKit/
├─ core/                 # Shared C++: pre/post, coordinates, embedding cache
│  ├─ include/
│  ├─ src/
│  └─ tests/
├─ runtime/
│  ├─ apple/             # iOS Core ML (Swift + ObjC++ bridge)
│  └─ android/           # Android TFLite (Kotlin + JNI/C++)
├─ ui/
│  ├─ ios/               # SamView (SwiftUI/UIKit)
│  └─ android/           # SamView (Jetpack Compose/View)
├─ models/
│  ├─ manifests/         # JSON metadata for converted models
│  └─ converters/        # PyTorch→ONNX→CoreML/TFLite scripts
├─ samples/
│  ├─ ios-sample/
│  └─ android-sample/
└─ tools/
   ├─ bench/             # latency/memory measurement
   └─ verify/            # numerical parity vs Python
```
- **core/** centralizes pre/post logic and coordinate math for both OSes.  
- **runtime/apple** hosts Core ML encoder/decoder; **runtime/android** wraps TFLite interpreters.  
- **ui/** provides SamView and a controller that debounces user inputs and triggers inference.

---

## 3. Public API (Parity-Oriented)
### 3.1 Conceptual Types
```txt
SamPoint(x, y, label)   // label: 1=positive, 0=negative
SamBox(x0, y0, x1, y1)  // image-space (left, top, right, bottom)
SamOptions {
  multimask_output: Bool = true
  return_logits:    Bool = false
  mask_threshold:   Float = 0.0 // 0 → no binarization; UI thresholds in real time
  max_masks:        Int   = 3
}
SamMask {
  width, height
  logits: Float[]   // optional; kept low-res then upsampled
  alpha:  UInt8[]   // 0-255 (binary or soft alpha)
  score:  Float     // decoder score/IoU
}
SamResult { masks: [SamMask] }
```

### 3.2 Session APIs
#### iOS (Swift)
```swift
public final class SamSession {
    public init(model: SamModelRef, config: RuntimeConfig = .bestAvailable) throws
    public func setImage(_ image: CGImage) throws
    public func predict(points: [SamPoint], box: SamBox?, maskInput: SamMaskRef?, options: SamOptions) async throws -> SamResult
    public func clear() // drop cached embedding
}
```

#### Android (Kotlin)
```kotlin
class SamSession(model: SamModelRef, config: RuntimeConfig = RuntimeConfig.Best) {
    suspend fun setImage(bitmap: Bitmap)
    suspend fun predict(points: List<SamPoint>, box: SamBox?, maskInput: SamMaskRef?, options: SamOptions): SamResult
    fun clear()
}
```

---

## 4. Pre/Post & Coordinate Transforms (Parity Critical)
- **Preprocess**:
  1) Given `orig_w x orig_h`, scale so **long side = S (e.g., 1024)**.  
  2) Place into **S x S** square canvas with symmetric padding.  
  3) Convert to **RGB** and normalize using model’s mean/std.
- **Coordinate Mapping**:
  - `scale = S / max(orig_w, orig_h)`  
  - `padX = (S - scaled_w) / 2`, `padY = (S - scaled_h) / 2`  
  - **toModel**: `xm = x*scale + padX`, `ym = y*scale + padY`  
  - **toImage**: `x = (xm - padX)/scale`, `y = (ym - padY)/scale`
- **Postprocess**:
  - Decoder’s low-res mask/logits → upsample to `S x S` → depad → resize to `orig_w x orig_h`.  
  - If `mask_threshold > 0`, binarize; otherwise keep logits/soft mask.  
  - With **multimask_output**, return sorted candidates.

---

## 5. Model Conversion Pipeline
> Deliver **separated** Encoder (image embedding) and Decoder (prompt→mask) for iOS and Android.

### 5.1 Steps
1. **PyTorch → ONNX** → `sam_encoder.onnx`, `sam_decoder.onnx`  
   - Static shapes for S=1024 where feasible.  
   - Decoder inputs: point_coords/labels, box_embeddings, mask_input, image_embeddings.
2. **ONNX → Core ML** (iOS)  
   - Use `coremltools`; convert to **FP16**; configure `MLModelConfiguration.computeUnits = .all`.
3. **ONNX → TFLite** (Android)  
   - Prefer **MobileSAM-class** models for op compatibility.  
   - Quantization: **FP16** (widely compatible), **INT8** (with calibration if possible).  
   - Produce `encoder.tflite` and `decoder.tflite` separately.

### 5.2 Manifest (`manifest.json`)
```json
{
  "name": "mobile_sam_v1",
  "input_size": 1024,
  "encoder": { "type": "coreml|tflite", "path": "encoder.mlmodelc|encoder.tflite", "precision": "fp16" },
  "decoder": { "type": "coreml|tflite", "path": "decoder.mlmodelc|decoder.tflite", "precision": "fp16" },
  "normalize": { "mean": [123.675,116.28,103.53], "std": [58.395,57.12,57.375] },
  "hash": { "encoder": "…", "decoder": "…" }
}
```

---

## 6. iOS Implementation (Core ML)
### 6.1 Runtime
- Package encoder/decoder as separate `.mlmodelc`. Bundle or lazy-download.  
- **Flow**:  
  - `setImage`: run **Encoder** → cache `image_embedding`.  
  - `predict`: build prompt tensors → run **Decoder** → postprocess results.
- **Optimizations**
  - `MLModelConfiguration.computeUnits = .all` (ANE/GPU/CPU).  
  - Reuse `CVPixelBuffer`/`MLMultiArray`.  
  - Keep embedding as **FP16**; repeated `predict` is then lightweight.

### 6.2 Swift Skeleton
```swift
let cfg = MLModelConfiguration()
cfg.computeUnits = .all
let encoder = try Encoder(configuration: cfg)
let decoder = try Decoder(configuration: cfg)

try session.setImage(cgImage) // preprocess → encoder → cache embedding
let result = try await session.predict(
  points: [...], box: ..., maskInput: nil,
  options: .init(multimaskOutput: true, returnLogits: true, maskThreshold: 0)
)
```

---

## 7. Android Implementation (TFLite)
### 7.1 Runtime
- Create TFLite `Interpreter` for **encoder** and **decoder** separately.  
- **Delegates**: prefer GPU Delegate → NNAPI → CPU.  
- Use `ByteBuffer.allocateDirect` for reusable input/output buffers.  
- **Flow**:  
  - `setImage`: `encoder.run()` → cache `image_embedding`.  
  - `predict`: encode prompt → `decoder.run()` → postprocess masks.

### 7.2 Kotlin Skeleton
```kotlin
val opts = Interpreter.Options().apply {
    // addDelegate(GpuDelegate()) // if available
    // setUseNNAPI(true)          // fallback strategy
    setNumThreads(Runtime.getRuntime().availableProcessors())
}
val encoder = Interpreter(loadModelFile("encoder.tflite"), opts)
val decoder = Interpreter(loadModelFile("decoder.tflite"), opts)

session.setImage(bitmap) // preprocess → encoder → cache
val res = session.predict(
  points, box, null,
  SamOptions(multimaskOutput = true, returnLogits = true)
)
```

---

## 8. SamView UI Specification
- **Inputs**: taps (+/-), drag-to-box, pinch-zoom, pan.  
- **Outputs**: semi-transparent mask overlays for multi-candidate results, threshold slider, candidate selection, boolean ops (add/subtract/intersect), Undo/Redo.  
- **Debounce**: trigger `predict` after **~60–80 ms** idle.  
- When **return_logits = true**, apply threshold entirely in UI (no re-inference).

- **iOS (SwiftUI)**: `SamController (ObservableObject)` manages state and invokes `SamSession`. Prefer Metal-backed overlay for blending.  
- **Android (Compose)**: `SamViewModel` + `Canvas` drawing; gestures via `pointerInput`.

---

## 9. AI Agent Work Plan (Checklist)
### 9.1 Repository Bootstrap
- [ ] Scaffold `SAMKit` per §2.  
- [ ] Configure CMake (core/), SPM (iOS), Gradle (Android).  
- [ ] CI: Xcodebuild (iOS) / Gradle (Android).

### 9.2 Pre/Post & Coordinates (core/ C++)
- [ ] Implement Resize+Pad(S), normalization, `toModel()`/`toImage()`.  
- [ ] Implement mask upsample → depad → resize.  
- [ ] Unit tests for geometry consistency.

### 9.3 Model Conversion (models/converters/)
- [ ] PyTorch→ONNX split export (Encoder/Decoder).  
- [ ] ONNX→Core ML (FP16). Produce `encoder.mlmodel`, `decoder.mlmodel`.  
- [ ] ONNX→TFLite (FP16/INT8). Produce `encoder.tflite`, `decoder.tflite`.  
- [ ] Emit `manifest.json`. Verify SHA256.

### 9.4 iOS Runtime (runtime/apple/)
- [ ] Core ML wrapper (`SamSession`) + embedding cache.  
- [ ] Map I/O to `CVPixelBuffer`/`MLMultiArray` with reuse.  
- [ ] iOS sample app: interactive test.

### 9.5 Android Runtime (runtime/android/)
- [ ] TFLite wrapper (`SamSession`) + GPU/NNAPI selection.  
- [ ] Reusable `ByteBuffer` I/O + embedding cache.  
- [ ] Android sample app: interactive test.

### 9.6 SamView (ui/)
- [ ] SwiftUI/Compose gestures + drawing.  
- [ ] Debounce + progress indicator.  
- [ ] Candidate switching, boolean ops, PNG/RLE export.

### 9.7 Verification (tools/verify/)
- [ ] Compare with Python `SamPredictor`: logits MSE & IoU.  
- [ ] Benchmarks: `setImage`/`predict` latency, memory.

### 9.8 Distribution
- [ ] iOS: **Swift Package** (binary or source + model downloader).  
- [ ] Android: **AAR** (Maven) + model downloader.  
- [ ] Versioning & CHANGELOG.

---

## 10. Benchmark Plan
- **Scenarios**: resolutions (512/768/1024), prompt counts (0–5 points + 1 box), multimask on/off.  
- **Metrics**: `setImage(ms)`, `predict(ms)`, P95, memory peak, optional power.  
- **Devices**: iPhone A15/16/17, Pixel 6/7/8/9, mid-tier Snapdragon.

---

## 11. Risks & Edge Cases
- **TFLite op gaps**: prioritize MobileSAM-like models; otherwise simplify decoder.  
- **Memory pressure**: large embeddings at 1024; mitigate with FP16/tiling/smaller profiles.  
- **Heavy UI interaction**: debounce & cancel in-flight jobs (latest input wins).  
- **Coordinate drift**: persist `(scale, padX, padY)` and always invert correctly.

---

## 12. Code Snippets (Selected)
### 12.1 iOS: `SamSession.setImage`
```swift
func setImage(_ image: CGImage) throws {
    let (tensor, meta) = Preprocessor.resizePadAndNormalize(image, target: 1024)
    self.meta = meta
    self.embedding = try encoder.predict(input: tensor) // cache FP16 embedding
}
```

### 12.2 Android: Decoder Call
```kotlin
val prompt = preprocessor.encodePrompt(points, box, maskInput, meta)
decoder.runForMultipleInputsOutputs(arrayOf(prompt.inputs...), mapOf(0 to outputMaskBuffer))
val mask = postprocessor.toMask(outputMaskBuffer, meta, options)
```

---

## 13. Output & Export
- **PNG (alpha)**, **RLE**, **COCO polygon**.  
- Keep logits for instant thresholding in UI without re-inference.

---

## 14. Implementation Order (MVP → Extended)
1) MVP with **MobileSAM**-class model at **1024/FP16** (iOS/Core ML & Android/TFLite).  
2) UI: points/box, multimask, threshold slider, export.  
3) Device optimizations (ANE/GPU/NNAPI).  
4) Extras: **scribble prompts**, **tiling for large images**, **HQ refiner** (optional).

---

## 15. Definition of Done
- Numerical parity vs Python under same settings: **IoU ≥ 0.95** target (account for quantization).  
- iPhone 14/15 target: `predict` **P95 < 40 ms** (1 point + 1 box, FP16, 1024).  
- Pixel 7/8 target: `predict` **P95 < 60 ms** (GPU/NNAPI, FP16, 1024).  
- Sample apps: stable for **1 minute continuous interaction** (no crashes/leaks).

---

## 16. Practical Notes
- Keep **encoder/decoder split**; after `setImage`, iterate decoder only.  
- Prefer `return_logits=true`; perform thresholding/smoothing on UI for responsiveness.  
- Use GPU for overlay blending (Metal / RenderEffect).

---

*End of document.*
