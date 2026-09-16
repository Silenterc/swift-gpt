//
//  TransformerBlockTests.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

import Testing
import MLX

@testable import SwiftGPT

let transformerConfig = GPTConfig(
    vocabSize: 100,
    contextLength: 16,
    embeddingDimension: 8,
    numHeads: 2,
    numLayers: 2,
    dropoutRate: 0.0,
    qkvBias: false
)

@Test("TransformerBlock preserves batched dimensions")
func transformerBlockBatchedDimensions() async throws {
    let block = TransformerBlock(config: transformerConfig)

    let input = MLXArray.zeros([2, 4, 8])

    let output = block(input)

    #expect(output.shape == [2, 4, 8])
}

@Test("TransformerBlock supports unbatched input")
func transformerBlockUnbatchedDimensions() async throws {
    let block = TransformerBlock(config: transformerConfig)

    let input = MLXArray.zeros([4, 8])

    let output = block(input)

    #expect(output.shape == [4, 8])
}
