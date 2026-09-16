//
//  GPTModelTests.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

import Testing
import MLX

@testable import SwiftGPT

let gptConfig = GPTConfig(
    vocabSize: 32,
    contextLength: 8,
    embeddingDimension: 8,
    numHeads: 2,
    numLayers: 2,
    dropoutRate: 0.0,
    qkvBias: false
)

@Test("GPTModel returns correct output dimensions")
func gptModelDimensions() async throws {
    let model = GPTModel(config: gptConfig)

    let inputIds = MLXArray(
        [
            UInt32(1), 2, 3, 4,
            5, 6, 7, 8
        ],
        [2, 4]
    )

    let output = model(inputIds)

    #expect(output.shape == [2, 4, gptConfig.vocabSize])
}

@Test("GPTModel fails for non-batched input")
func gptModelFailsForNonBatchedInput() async throws {
    await #expect(processExitsWith: .failure) {
        let model = GPTModel(config: gptConfig)

        let inputIds = MLXArray(
            [UInt32(1), 2, 3, 4]
        )

        _ = model(inputIds)
    }
}

@Test("GPTModel fails when sequence exceeds context length")
func gptModelFailsForLongSequence() async throws {
    await #expect(processExitsWith: .failure) {
        let model = GPTModel(config: gptConfig)

        let inputIds = MLXArray(
            [
                UInt32(1), 2, 3, 4, 5,
                6, 7, 8, 9
            ],
            [1, 9]
        )

        _ = model(inputIds)
    }
}

@Test("GPTModel has expected parameter count")
func gptModelParameterCount() async throws {
    let config = GPTConfig.gpt2Small
    let model = GPTModel(config: config)

    let parameterCount = model.trainableParameters()
        .flattenedValues()
        .map { $0.size }
        .reduce(0, +)

    #expect(parameterCount == 163_009_536) // Or 124M with weight tying
    
    // GPT-2 ties the token embedding and output projection weights
    // Our model currently keeps them separate, so exclude the output
    let outputProjectionParameters = config.embeddingDimension * config.vocabSize

    let parametersWithWeightTying = parameterCount - outputProjectionParameters

    #expect(parametersWithWeightTying == 124_412_160)
}
