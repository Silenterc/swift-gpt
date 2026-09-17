//
//  GPTConfig.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

public struct GPTConfig: Sendable {
    public let vocabSize: Int
    public let contextLength: Int
    public let embeddingDimension: Int
    public let numHeads: Int
    public let numLayers: Int
    public let dropoutRate: Float
    public let qkvBias: Bool
}

public extension GPTConfig {
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
