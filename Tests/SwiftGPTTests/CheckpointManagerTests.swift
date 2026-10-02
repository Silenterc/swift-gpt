//
//  CheckpointManagerTests.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 02.10.2026.
//

import Foundation
import Testing
import MLX
import MLXNN

@testable import SwiftGPT

@Test("Test save checkpoint")
func saveCheckpoint() throws {
    let testDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)

    defer {
        try? FileManager.default.removeItem(at: testDir)
    }

    let loader = try makeDataLoader(in: testDir)

    let module = Linear(2, 3)
    let checkpointDir = testDir.appendingPathComponent("checkpoint")

    try CheckpointManager.save(
        module: module,
        dataLoader: loader,
        globalStep: 42,
        to: checkpointDir
    )

    #expect(
        FileManager.default.fileExists(
            atPath: checkpointDir
                .appendingPathComponent("weights.safetensors")
                .path
        )
    )

    #expect(
        FileManager.default.fileExists(
            atPath: checkpointDir
                .appendingPathComponent("state.json")
                .path
        )
    )
}

@Test("Test saved weights match module weights")
func savedWeightsMatch() throws {
    let testDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)

    defer {
        try? FileManager.default.removeItem(at: testDir)
    }

    let loader = try makeDataLoader(in: testDir)

    let module = Linear(2, 3)
    let checkpointDir = testDir.appendingPathComponent("checkpoint")

    try CheckpointManager.save(
        module: module,
        dataLoader: loader,
        globalStep: 42,
        to: checkpointDir
    )

    let weightsURL = checkpointDir
        .appendingPathComponent("weights.safetensors")

    let savedWeights = try MLX.loadArrays(url: weightsURL)

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

@Test("Test load checkpoint restores weights")
func loadCheckpointWeights() throws {
    let testDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)

    defer {
        try? FileManager.default.removeItem(at: testDir)
    }

    let sourceLoader = try makeDataLoader(in: testDir)
    let sourceModule = Linear(2, 3)

    let checkpointDir = testDir.appendingPathComponent("checkpoint")

    try CheckpointManager.save(
        module: sourceModule,
        dataLoader: sourceLoader,
        globalStep: 42,
        to: checkpointDir
    )

    let targetLoader = try makeDataLoader(in: testDir)
    let targetModule = Linear(2, 3)

    _ = try CheckpointManager.load(
        module: targetModule,
        dataLoader: targetLoader,
        from: checkpointDir
    )

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

@Test("Test load checkpoint restores training state")
func loadCheckpointState() throws {
    let testDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)

    defer {
        try? FileManager.default.removeItem(at: testDir)
    }

    let sourceLoader = try makeDataLoader(in: testDir)

    // Advance the loader so its index is no longer zero.
    _ = try sourceLoader.nextBatch()

    let savedIndex = sourceLoader.getCurrentIndex()
    let savedGlobalStep = 123

    let module = Linear(2, 3)
    let checkpointDir = testDir.appendingPathComponent("checkpoint")

    try CheckpointManager.save(
        module: module,
        dataLoader: sourceLoader,
        globalStep: savedGlobalStep,
        to: checkpointDir
    )

    let restoredLoader = try makeDataLoader(in: testDir)

    let restoredGlobalStep = try CheckpointManager.load(
        module: module,
        dataLoader: restoredLoader,
        from: checkpointDir
    )

    #expect(restoredGlobalStep == savedGlobalStep)
    #expect(restoredLoader.getCurrentIndex() == savedIndex)
}

private func makeDataLoader(in testDir: URL) throws -> DataLoader {
    let tokenDir = testDir.appendingPathComponent("tokens")

    let writer = try TokenFileWriter(
        outputDirectory: tokenDir,
        maxTokensPerShard: 10_000
    )

    let tokens = (0..<100).map(UInt32.init)

    try writer.write(tokens)
    try writer.finish()

    let dataset = try TokenDataset(
        tokenDirectory: tokenDir
    )

    return DataLoader(
        dataset: dataset,
        batchSize: 2,
        maxLength: 2,
        stride: 2
    )
}
