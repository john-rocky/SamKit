// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

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
        ),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
    ],
    targets: [
        .target(
            name: "SAMKit",
            dependencies: [],
            path: "Sources/SAMKit"
        ),
        .target(
            name: "SAMKitUI",
            dependencies: ["SAMKit"],
            path: "Sources/SAMKitUI"
        ),
        .testTarget(
            name: "SAMKitTests",
            dependencies: ["SAMKit"],
            path: "Tests"
        ),
    ]
)