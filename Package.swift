// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// Swift Package Manager manifest for the Apple (Core ML) runtime. The shared C++ core
// under `core/` is built separately via CMake (CMakeLists.txt) and is not required by the
// Swift targets, so it is intentionally not part of this SwiftPM graph.
let package = Package(
    name: "SAMKit",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "SAMKit",
            targets: ["SAMKit"]
        ),
        .library(
            name: "SAMKitGrounding",
            targets: ["SAMKitGrounding"]
        ),
        .library(
            name: "SAMKitUI",
            targets: ["SAMKitUI"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "SAMKit",
            dependencies: [],
            path: "runtime/apple/Sources/SAMKit"
        ),
        .target(
            name: "SAMKitGrounding",
            dependencies: ["SAMKit"],
            path: "runtime/apple/Sources/SAMKitGrounding"
        ),
        .target(
            name: "SAMKitUI",
            dependencies: ["SAMKit", "SAMKitGrounding"],
            path: "runtime/apple/Sources/SAMKitUI"
        ),
    ]
)
