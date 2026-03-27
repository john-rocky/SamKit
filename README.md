# SAMKit

A high-performance iOS library for running [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) on-device with Core ML.

## Features

- **Point & Box Segmentation** - Tap or draw a box to segment any object
- **Text-Based Detection** - Describe objects in natural language (YOLO-World + CLIP)
- **Object Extraction** - Lift objects with transparent background, copy/save/share
- **Two SAM Models** - MobileSAM (fast) and SAM2 Tiny (accurate), switchable at runtime
- **Pre-built UI Components** - Drop-in SwiftUI views for segmentation workflows
- **Optimized for Mobile** - Neural Engine / GPU acceleration, FP16 inference

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

Three library products are available:

| Product | Description |
|---------|-------------|
| `SAMKit` | Core inference - point/box segmentation |
| `SAMKitGrounding` | Text-based detection (YOLO-World + CLIP) |
| `SAMKitUI` | Pre-built SwiftUI components |

### 2. Download Model Files

Model files are distributed via [GitHub Releases](https://github.com/john-rocky/SamKit/releases).

Download the `.mlpackage` files and add them to your Xcode project (make sure "Copy items if needed" is checked and they are added to your app target).

#### MobileSAM (required, ~23 MB)

| File | Size |
|------|------|
| `mobile_sam_encoder.mlpackage` | 13 MB |
| `mobile_sam_decoder.mlpackage` | 9.8 MB |
| `mobile_sam_prompt_encoder_weights.json` | 40 KB |

#### SAM2 Tiny (optional, ~76 MB)

| File | Size |
|------|------|
| `SAM2TinyImageEncoderFLOAT16.mlpackage` | 64 MB |
| `SAM2TinyPromptEncoderFLOAT16.mlpackage` | 2.0 MB |
| `SAM2TinyMaskDecoderFLOAT16.mlpackage` | 9.8 MB |

#### Grounding Models (optional, for text-based detection, ~148 MB)

| File | Size |
|------|------|
| `clip_text_encoder.mlpackage` | 121 MB |
| `yoloworld_detector.mlpackage` | 25 MB |
| `clip_vocab.json` | 1.6 MB |
| `cv4_params.json` | 4 KB |

## Quick Start

### Point / Box Segmentation

```swift
import SAMKit

// Load MobileSAM
let model = try SamModelRef.bundled(.mobileSam)
let session = try SamSession(model: model, config: .bestAvailable)

// Set image (encoder runs once, result is cached)
try session.setImage(cgImage)

// Segment with a point
let result = try session.predict(
    points: [SamPoint(x: 100, y: 200, label: .positive)]
)

// Use the best mask
let mask = result.masks.first!
// mask.cgImage  - grayscale mask image
// mask.alpha    - raw alpha data
// mask.score    - IoU confidence score
```

### SAM2 Tiny

```swift
import SAMKit

// Load SAM2 Tiny (3-component model)
let session = try Sam2Session(modelName: "SAM2Tiny", config: .bestAvailable)

try session.setImage(cgImage)
let result = try session.predict(
    points: [SamPoint(x: 100, y: 200, label: .positive)]
)
```

### Text-Based Detection + Segmentation

```swift
import SAMKit
import SAMKitGrounding

let groundingModel = try GroundingModelRef.bundled()
let samModel = try SamModelRef.bundled(.mobileSam)
let session = try TextSegmentationSession(
    groundingModel: groundingModel,
    samModel: samModel
)

try session.setImage(cgImage)
let result = try session.predict(text: "dog, cat")

// result contains detected bounding boxes + segmentation masks
```

## Architecture

```
SAMKit/
├── core/              # Shared C++ preprocessing/postprocessing
├── runtime/apple/     # iOS Core ML implementation
│   ├── SAMKit/        #   Core inference
│   ├── SAMKitGrounding/ # Text detection (YOLO-World + CLIP)
│   └── SAMKitUI/      #   SwiftUI components
├── models/            # Conversion scripts and manifests
├── samples/           # Demo application
│   └── ios-sample/    #   Full-featured iOS demo
└── ui/ios/            # UI component sources
```

## Sample App

The demo app in `samples/ios-sample/` demonstrates all features:

1. Clone the repo and open `SAMKitDemo.xcodeproj`
2. Download model files from [Releases](https://github.com/john-rocky/SamKit/releases) and add them to the project
3. Build and run on a physical device

## Model Conversion

To convert models from PyTorch yourself:

```bash
cd models/converters
pip install -r requirements.txt

# MobileSAM
python convert_to_coreml.py --model mobile_sam

# SAM2 Tiny (HuggingFace)
python convert_sam2_to_coreml.py
```

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything)
- [SAM 2 (Meta AI)](https://github.com/facebookresearch/sam2)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
