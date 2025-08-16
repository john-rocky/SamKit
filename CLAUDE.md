# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAMKit is a mobile library for implementing Meta's Segment Anything Model (SAM) on iOS and Android platforms. The project is currently in the planning phase with a comprehensive strategy document (`SAMKit_Strategy_CoreML_TFLite_EN.md`).

## Current Status

**Important**: This project is in the initial planning stage. No code has been implemented yet. The repository contains only the strategy document.

## Planned Architecture

The project will implement a cross-platform mobile library with:
- **Core C++ layer**: Shared preprocessing/postprocessing logic
- **iOS runtime**: Core ML-based implementation
- **Android runtime**: TensorFlow Lite-based implementation
- **UI components**: Native interfaces for both platforms

### Directory Structure (Planned)
```
SAMKit/
├─ core/           # Shared C++ code for pre/post processing
├─ runtime/        
│  ├─ apple/       # iOS Core ML implementation
│  └─ android/     # Android TFLite implementation
├─ ui/
│  ├─ ios/         # SwiftUI/UIKit interface
│  └─ android/     # Jetpack Compose interface
├─ models/         # Model conversion scripts and manifests
├─ samples/        # Sample applications
└─ tools/          # Benchmarking and verification
```

## Key Technologies

### iOS Platform
- Core ML for model inference
- Swift/SwiftUI for UI
- Metal for GPU acceleration
- Swift Package Manager for distribution

### Android Platform
- TensorFlow Lite for model inference
- Kotlin/Jetpack Compose for UI
- GPU Delegate/NNAPI for acceleration
- Gradle build system

### Shared Core
- C++ for cross-platform logic
- CMake build system

## Model Pipeline

SAM model conversion flow:
1. PyTorch checkpoint → ONNX
2. ONNX → Core ML (iOS) / TensorFlow Lite (Android)
3. Separate encoder and decoder models for efficiency

## Implementation Priorities

When implementing features, follow the strategy document's phased approach:
1. Core C++ preprocessing/postprocessing
2. Model conversion pipeline
3. iOS runtime implementation
4. Android runtime implementation
5. UI components
6. Verification and benchmarking tools

## Performance Considerations

- Image encoder runs once per image (cached)
- Prompt encoder + mask decoder run per interaction
- Target: <100ms inference on modern mobile devices
- Use platform-specific optimizations (Metal, GPU Delegate)

## Testing Strategy

When tests are implemented:
- Unit tests for core C++ logic
- Platform-specific runtime tests
- UI component tests
- End-to-end integration tests
- Performance benchmarks

## Build Commands (To Be Implemented)

### iOS
```bash
# Build will use Swift Package Manager
swift build
swift test
```

### Android
```bash
# Build will use Gradle
./gradlew build
./gradlew test
```

### Core C++
```bash
# Build will use CMake
cmake -B build
cmake --build build
ctest --test-dir build
```

## Important Notes

1. The project follows a modular architecture to maximize code reuse between platforms
2. Model files should be optimized for mobile (quantization, pruning)
3. Memory management is critical for mobile deployment
4. Follow platform-specific best practices for iOS and Android
5. Maintain API compatibility across platforms where possible