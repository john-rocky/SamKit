# SAMKit

A high-performance mobile library for running Segment Anything Model (SAM) on iOS and Android devices.

## Features

- **iOS Support**: SAM 2.1 and MobileSAM via Core ML
- **Android Support**: MobileSAM via TensorFlow Lite  
- **Optimized Performance**: <40ms inference on iPhone 14/15, <60ms on Pixel 7/8
- **Interactive UI**: Point prompts, box selection, multi-mask output
- **Cross-platform Core**: Shared C++ preprocessing/postprocessing

## Architecture

```
SAMKit/
├── core/           # Shared C++ preprocessing/postprocessing
├── runtime/        
│   ├── apple/      # iOS Core ML implementation
│   └── android/    # Android TFLite implementation
├── ui/
│   ├── ios/        # SwiftUI interface
│   └── android/    # Jetpack Compose interface
├── models/         # Model conversion and manifests
├── samples/        # Sample applications
└── tools/          # Benchmarking and verification
```

## Supported Models

### iOS (Core ML)
- **SAM 2.1**: Official SAM 2.1 models (ViT-B, ViT-L, ViT-H)
- **MobileSAM**: Optimized for mobile devices

### Android (TensorFlow Lite)
- **MobileSAM**: TFLite-compatible lightweight model

## Requirements

### iOS
- iOS 15.0+
- Xcode 14.0+
- Swift 5.7+

### Android
- Android API 26+ (Android 8.0)
- Android Studio Arctic Fox+
- Kotlin 1.8+

## Installation

### iOS (Swift Package Manager)

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/SAMKit.git", from: "1.0.0")
]
```

### Android (Gradle)

```gradle
dependencies {
    implementation 'com.samkit:samkit:1.0.0'
    implementation 'com.samkit:samkit-ui:1.0.0'
}
```

## Quick Start

### iOS

```swift
import SAMKit

// Initialize session
let session = try SamSession(model: .sam2_1_base, config: .bestAvailable)

// Set image
try session.setImage(cgImage)

// Run inference
let result = try await session.predict(
    points: [SamPoint(x: 100, y: 200, label: 1)],
    box: nil,
    maskInput: nil,
    options: SamOptions(multimaskOutput: true)
)

// Get masks
for mask in result.masks {
    // Process mask.alpha or mask.logits
}
```

### Android

```kotlin
import com.samkit.SamSession

// Initialize session
val session = SamSession(model = SamModel.MOBILE_SAM)

// Set image
session.setImage(bitmap)

// Run inference
val result = session.predict(
    points = listOf(SamPoint(100f, 200f, 1)),
    box = null,
    maskInput = null,
    options = SamOptions(multimaskOutput = true)
)

// Get masks
result.masks.forEach { mask ->
    // Process mask.alpha or mask.logits
}
```

## Building from Source

### Prerequisites

- CMake 3.20+
- Python 3.8+ (for model conversion)
- For iOS: Xcode 14+
- For Android: Android NDK r23+

### Build Steps

```bash
# Clone repository
git clone https://github.com/yourusername/SAMKit.git
cd SAMKit

# Build Core C++
mkdir build && cd build
cmake ..
make

# iOS
cd runtime/apple
swift build

# Android
cd runtime/android
./gradlew build
```

## Model Conversion

Convert PyTorch models to Core ML/TFLite:

```bash
cd models/converters

# For iOS (Core ML)
python convert_to_coreml.py \
    --model sam2_hiera_base_plus \
    --output ../manifests/sam2_1_base.mlpackage

# For Android (TFLite)
python convert_to_tflite.py \
    --model mobile_sam \
    --output ../manifests/mobile_sam.tflite
```

## Performance

| Device | Model | setImage (ms) | predict (ms) | Memory (MB) |
|--------|-------|---------------|--------------|-------------|
| iPhone 15 Pro | SAM 2.1 Base | 150 | 35 | 380 |
| iPhone 14 | MobileSAM | 80 | 25 | 180 |
| Pixel 8 Pro | MobileSAM | 120 | 55 | 200 |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Meta AI for the original [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) team for the mobile-optimized variant

## Citation

If you use SAMKit in your research, please cite:

```bibtex
@software{samkit2024,
  title = {SAMKit: Mobile Segment Anything},
  year = {2024},
  url = {https://github.com/yourusername/SAMKit}
}
```