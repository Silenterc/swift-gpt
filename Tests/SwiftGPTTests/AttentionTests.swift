//
//  AttentionTests.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 02.09.2026.
//

import Testing
import SwiftGPT
import MLX


@Test("Self attention succeeds and the dimensions are correct")
func selfAttentionDimensions() async throws {
    let attention = SelfAttention(inputDimensions: 4, outputDimensions: 8)
    // 6 tokens, embedding dimension is 4
    let x = MLXRandom.uniform(low: 0, high: 1, [6, 4])
    let output = attention(x)
    #expect(output.shape == [6, 8])
}
