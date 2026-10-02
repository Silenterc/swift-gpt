//
//  WeightManagerTests.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 02.10.2026.
//

import Foundation
import Testing
import MLX
import MLXNN

@testable import SwiftGPT

@Test("Test save weights")
func saveWeights() throws {
    let module = Linear(2, 3)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(UUID()).safetensors")

    defer {
        try? FileManager.default.removeItem(at: url)
    }

    try WeightManager.save(module: module, to: url)

    #expect(FileManager.default.fileExists(atPath: url.path))
}

@Test("Test saved weights match module weights")
func savedWeightsMatch() throws {
    let module = Linear(2, 3)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(UUID()).safetensors")

    defer {
        try? FileManager.default.removeItem(at: url)
    }

    try WeightManager.save(module: module, to: url)

    let savedWeights = try MLX.loadArrays(url: url)
    let moduleWeights = Dictionary(
        uniqueKeysWithValues: module.parameters().flattened()
    )

    #expect(savedWeights.keys == moduleWeights.keys)

    for key in moduleWeights.keys {
        #expect(
            allClose(
                savedWeights[key]!,
                moduleWeights[key]!
            ).item(Bool.self)
        )
    }
}

@Test("Test load weights")
func loadWeights() throws {
    let sourceModule = Linear(2, 3)
    let targetModule = Linear(2, 3)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(UUID()).safetensors")

    defer {
        try? FileManager.default.removeItem(at: url)
    }

    try WeightManager.save(module: sourceModule, to: url)
    try WeightManager.load(module: targetModule, from: url)

    let sourceWeights = Dictionary(
        uniqueKeysWithValues: sourceModule.parameters().flattened()
    )

    let targetWeights = Dictionary(
        uniqueKeysWithValues: targetModule.parameters().flattened()
    )

    #expect(sourceWeights.keys == targetWeights.keys)

    for key in sourceWeights.keys {
        #expect(
            allClose(
                sourceWeights[key]!,
                targetWeights[key]!
            ).item(Bool.self)
        )
    }
}
