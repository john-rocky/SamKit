<p align="center">
  <img src="https://github.com/user-attachments/assets/d25ff3c2-91e1-4c86-8959-ff7b7dc3829f" width="300" alt="SAMKit" />
</p>
<p align="center">
  <strong>Segment Anything, right on your iPhone.</strong>
</p>
<p align="center">
  <a href="#installation">Install</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#sample-app">Demo App</a> &middot;
  <a href="https://github.com/john-rocky/SamKit/releases">Download Models</a>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2a4364ee-a3fc-4e40-a0bb-b3c7c9bfa0f5" width="300" alt="SAMKit Demo" />
</p>

---

SAMKit brings Meta's [Segment Anything Model](https://github.com/facebookresearch/segment-anything) to iOS as a native Swift package. Tap, draw, or describe any object to instantly segment it — all inference runs on-device with Core ML, no server required.

## Features

- **Point & Box** &mdash; Tap a point or drag a bounding box to segment any object
- **Text Prompt** &mdash; Type `"dog"` or `"red cup"` to find and segment objects, powered by YOLO-World + CLIP
- **Subject Lift** &mdash; Long-press to lift the segmented object from the scene, then copy, save, or share as a transparent PNG
- **Two Backbones** &mdash; MobileSAM (fast, 23 MB) and SAM2 Tiny (accurate, 76 MB)
- **Drop-in UI** &mdash; Pre-built SwiftUI views for shipping a segmentation feature in minutes
- **Fully On-Device** &mdash; Neural Engine / GPU acceleration, FP16, zero network calls

## Requirements

- iOS 15.0+
- Xcode 14.0+
- Swift 5.7+

## Installation

### 1. Add the Swift Package

```swift
dependencies: [
    .package(url: "https://github.com/john-rocky/SamKit.git", from: "1.0.0")
]
```

| Product | What it does |
|---------|-------------|
| `SAMKit` | Core segmentation engine (point / box) |
| `SAMKitGrounding` | Open-vocabulary text detection (YOLO-World + CLIP) |
| `SAMKitUI` | Ready-made SwiftUI views |

### 2. Download Models

Grab the `.mlpackage` files from **[Releases](https://github.com/john-rocky/SamKit/releases)** and drag them into your Xcode project.

<details>
<summary><strong>MobileSAM</strong> &mdash; 23 MB (required)</summary>

| File | Size |
|------|------|
| `mobile_sam_encoder.mlpackage` | 13 MB |
| `mobile_sam_decoder.mlpackage` | 9.8 MB |
| `mobile_sam_prompt_encoder_weights.json` | 40 KB |
</details>

<details>
<summary><strong>SAM2 Tiny</strong> &mdash; 76 MB (optional)</summary>

| File | Size |
|------|------|
| `SAM2TinyImageEncoderFLOAT16.mlpackage` | 64 MB |
| `SAM2TinyPromptEncoderFLOAT16.mlpackage` | 2.0 MB |
| `SAM2TinyMaskDecoderFLOAT16.mlpackage` | 9.8 MB |
</details>

<details>
<summary><strong>Grounding (YOLO-World + CLIP)</strong> &mdash; 148 MB (optional)</summary>

| File | Size |
|------|------|
| `clip_text_encoder.mlpackage` | 121 MB |
| `yoloworld_detector.mlpackage` | 25 MB |
| `clip_vocab.json` | 1.6 MB |
| `cv4_params.json` | 4 KB |
</details>

## Quick Start

### Point & Box Segmentation

```swift
import SAMKit

let session = try SamSession(
    model: .bundled(.mobileSam),
    config: .bestAvailable
)

try session.setImage(cgImage)

let result = try session.predict(
    points: [SamPoint(x: 100, y: 200, label: .positive)]
)

let mask = result.masks.first!   // .cgImage, .alpha, .score
```

### SAM2 Tiny

```swift
import SAMKit

let session = try Sam2Session(
    modelName: "SAM2Tiny",
    config: .bestAvailable
)

try session.setImage(cgImage)
let result = try session.predict(
    points: [SamPoint(x: 100, y: 200, label: .positive)]
)
```

### Text-Prompted Segmentation

```swift
import SAMKit
import SAMKitGrounding

let session = try TextSegmentationSession(
    groundingModel: .bundled(),
    samModel: .bundled(.mobileSam)
)

try session.setImage(cgImage)
let result = try session.segment(query: "dog, cat")
// result.masks      — segmentation masks
// result.detections — bounding boxes + labels
```

### Subject Lifting

```swift
import SAMKit

// After segmentation, extract the object with transparency
let extracted = SamMask.extractObject(from: cgImage, masks: result.masks)
// Returns a CGImage with transparent background — ready for copy/save/share
```

## Architecture

```
SAMKit/
├── runtime/apple/
│   ├── SAMKit/            # Core inference engine
│   ├── SAMKitGrounding/   # YOLO-World + CLIP text detection
│   └── SAMKitUI/          # SwiftUI components
├── models/converters/     # PyTorch -> Core ML conversion scripts
├── samples/ios-sample/    # Full demo app
└── CLAUDE.md
```

## Sample App

```bash
git clone https://github.com/john-rocky/SamKit.git
open samples/ios-sample/SAMKitDemo.xcodeproj
```

Download models from [Releases](https://github.com/john-rocky/SamKit/releases), add to the project, and run on a physical device.

## Model Conversion

Convert from PyTorch checkpoints yourself:

```bash
cd models/converters
pip install -r requirements.txt

# MobileSAM
python convert_to_coreml.py --model mobile_sam

# SAM2 Tiny
python convert_sam2_to_coreml.py

# YOLO-World (S/M/L/X)
python convert_yoloworld_to_coreml.py --size s
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Segment Anything](https://github.com/facebookresearch/segment-anything) & [SAM 2](https://github.com/facebookresearch/sam2) — Meta AI
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) — Chaoning Zhang et al.
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) — Tencent AILab
- [OpenAI CLIP](https://github.com/openai/CLIP)
