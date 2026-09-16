//
//  FeedForwardTests.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

import Testing
import MLX

@testable import SwiftGPT

@Test("FeedForward preserves dimensions")
func feedForwardDimensions() async throws {
    let feedForward = FeedForward(
        embeddingDimension: 8
    )

    let input = MLXArray.zeros([2, 4, 8])

    let output = feedForward(input)

    #expect(output.shape == [2, 4, 8])
}
