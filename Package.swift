// swift-tools-version: 5.7
import PackageDescription

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
            name: "SAMKitUI",
            targets: ["SAMKitUI"]
        )
    ],
    dependencies: [],
    targets: [
        // Main SAMKit target
        .target(
            name: "SAMKit",
            dependencies: ["SAMKitCore"],
            path: "runtime/apple/Sources/SAMKit",
            resources: [
                .process("Resources")
            ],
            swiftSettings: [
                .define("ACCELERATE_NEW_LAPACK"),
                .define("ACCELERATE_LAPACK_ILP64")
            ]
        ),
        
        // Core C++ implementation
        .target(
            name: "SAMKitCore",
            dependencies: [],
            path: "core",
            sources: ["src"],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include"),
                .define("SAMKIT_BUILD", to: "1"),
                .standard(.cxx17)
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreML"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders")
            ]
        ),
        
        // UI Components
        .target(
            name: "SAMKitUI",
            dependencies: ["SAMKit"],
            path: "ui/ios/Sources",
            resources: [
                .process("Resources")
            ]
        ),
        
        // Tests
        .testTarget(
            name: "SAMKitTests",
            dependencies: ["SAMKit"],
            path: "runtime/apple/Tests",
            resources: [
                .process("Resources")
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
)