//
//  LayerNormTests.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

import Testing
import MLX

@testable import SwiftGPT

@Test("LayerNorm preserves dimensions")
func layerNormDimensions() async throws {
    let layerNorm = LayerNorm(embeddingDimension: 4)
    let input = MLXArray(
        [1.0, 2.0, 3.0, 4.0,
         5.0, 6.0, 7.0, 8.0] as [Float],
        [2, 4]
    )

    let output = layerNorm(input)

    #expect(output.shape == [2, 4])
}

@Test("LayerNorm normalizes embeddings")
func layerNormNormalization() async throws {
    let layerNorm = LayerNorm(embeddingDimension: 4)

    let input = MLXArray(
        [1.0, 2.0, 3.0, 4.0,
         2.0, 4.0, 6.0, 8.0] as [Float],
        [2, 4]
    )

    let output = layerNorm(input)

    let means = output.mean(axis: -1)
    let variances = output.variance(axis: -1)

    let expectedMeans = MLXArray.zeros([2])
    let expectedVariances = MLXArray.ones([2])

    #expect(
        MLX.allClose(means, expectedMeans, atol: 1e-5).item(Bool.self)
    )

    #expect(
        MLX.allClose(variances, expectedVariances, atol: 1e-4).item(Bool.self)
    )
}
