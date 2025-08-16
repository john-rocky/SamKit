# SAMKit Sample Applications

This directory contains sample applications demonstrating the use of SAMKit on iOS and Android platforms.

## iOS Sample

Located in `ios-sample/`, this SwiftUI application demonstrates:

- Image selection from photo library or camera
- Model selection (MobileSAM, SAM 2.1 variants)
- Interactive segmentation with point and box prompts
- Multi-mask visualization
- Export functionality

### Requirements
- iOS 15.0+
- Xcode 14.0+
- Swift 5.7+

### Setup
1. Open `SAMKitDemo.xcodeproj` in Xcode
2. Add SAMKit package dependency (already configured)
3. Download and add model files to the bundle
4. Build and run on device or simulator

### Features
- **Model Selection**: Choose between MobileSAM (fast) and SAM 2.1 models
- **Interactive UI**: Tap to add positive/negative points, drag to create boxes
- **Real-time Preview**: See segmentation results instantly
- **Export Options**: Save or share segmented images

## Android Sample

Located in `android-sample/`, this Jetpack Compose application demonstrates:

- Image selection from gallery or camera
- MobileSAM model integration
- Interactive segmentation UI
- Material You design

### Requirements
- Android API 26+ (Android 8.0)
- Android Studio Arctic Fox+
- Kotlin 1.8+

### Setup
1. Open the project in Android Studio
2. Add MobileSAM model files to `app/src/main/assets/models/`
3. Sync project with Gradle files
4. Build and run on device or emulator

### Features
- **Material Design 3**: Modern Android UI with dynamic colors
- **Permission Handling**: Camera and storage permissions
- **GPU Acceleration**: TensorFlow Lite GPU delegate support
- **Compose UI**: Fully built with Jetpack Compose

## Model Files

Both samples require model files to be added:

### iOS Models
Place Core ML models in the app bundle:
- `mobile_sam_encoder.mlmodelc`
- `mobile_sam_decoder.mlmodelc`
- `sam2_base_encoder.mlmodelc` (optional)
- `sam2_base_decoder.mlmodelc` (optional)

### Android Models
Place TFLite models in `assets/models/`:
- `mobile_sam_encoder.tflite`
- `mobile_sam_decoder.tflite`

## Converting Models

Use the provided conversion scripts:

```bash
# For iOS (Core ML)
cd models/converters
python convert_to_coreml.py --model mobile_sam --output ../manifests

# For Android (TFLite)
python convert_to_tflite.py --checkpoint path/to/mobile_sam.pth --output ../manifests
```

## Common Issues

### iOS
- **Model not found**: Ensure models are added to the Xcode project and included in the bundle
- **Memory warnings**: SAM models are large; test on real devices for accurate performance
- **Slow inference**: Use MobileSAM for better performance on older devices

### Android
- **TFLite errors**: Ensure you're using compatible model versions
- **OOM errors**: Enable `largeHeap` in manifest and use smaller input sizes
- **GPU delegate issues**: Fallback to CPU if GPU acceleration fails

## Performance Tips

1. **Use MobileSAM** for real-time interaction on mobile devices
2. **Optimize input size**: Start with 512x512 for testing, increase to 1024x1024 for production
3. **Cache embeddings**: The image encoder output can be reused for multiple prompts
4. **Debounce interactions**: Wait 60-80ms after user input before running inference
5. **GPU acceleration**: Enable GPU/Neural Engine for significant speedup

## License

These samples are provided under the same Apache 2.0 license as SAMKit.