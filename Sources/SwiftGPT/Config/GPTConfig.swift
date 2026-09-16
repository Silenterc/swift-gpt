//
//  GPTConfig.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

public struct GPTConfig: Sendable {
    let vocabSize: Int
    let contextLength: Int
    let embeddingDimension: Int
    let numHeads: Int
    let numLayers: Int
    let dropoutRate: Float
    let qkvBias: Bool
}

extension GPTConfig {
    /// 124M parameters
    static let gpt2Small = GPTConfig(
        vocabSize: 50_257, // BPE tokenizer
        contextLength: 1024,
        embeddingDimension: 768,
        numHeads: 12,
        numLayers: 12,
        dropoutRate: 0.1,
        qkvBias: false)
}
