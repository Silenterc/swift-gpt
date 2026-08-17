// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-gpt",
    platforms: [
        .macOS(.v14),
        .iOS(.v14),
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "swift-gpt",
            targets: ["SwiftGPT"]
        ),
        .executable(
            name: "train",
            targets: ["Train"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.4"), // Support Swift 6.2 for now
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", exact: "3.31.4")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "SwiftGPT",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ]
        ),
        .executableTarget(
            name: "Train",
            dependencies: [
                "SwiftGPT",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift")
                
            ]
        ),
        .testTarget(
            name: "SwiftGPTTests",
            dependencies: ["SwiftGPT"]
        ),
    ]
)
