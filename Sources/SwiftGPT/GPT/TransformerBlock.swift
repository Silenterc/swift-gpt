//
//  TransformerBlock.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

import MLX
import MLXNN

public class TransformerBlock: Module, UnaryLayer {
    @ModuleInfo var layerNorm1: LayerNorm
    @ModuleInfo var multiheadAttention: MultiheadAttention
    @ModuleInfo var dropoutShortcut: Dropout
    @ModuleInfo var layerNorm2: LayerNorm
    @ModuleInfo var feedForward: FeedForward
    
    public init(config: GPTConfig) {
        self.layerNorm1 = LayerNorm(embeddingDimension: config.embeddingDimension)
        self.multiheadAttention = MultiheadAttention(
            inputDimensions: config.embeddingDimension,
            outputDimensions: config.embeddingDimension,
            contextLength: config.contextLength,
            numHeads: config.numHeads,
            dropout: config.dropoutRate,
            bias: config.qkvBias
        )
        self.dropoutShortcut = Dropout(p: config.dropoutRate)
        self.layerNorm2 = LayerNorm(embeddingDimension: config.embeddingDimension)
        self.feedForward = FeedForward(embeddingDimension: config.embeddingDimension)
    }
    
    public func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        var input = x
        var shortcut = input
        
        input = self.layerNorm1(input)
        input = self.multiheadAttention(input)
        input = self.dropoutShortcut(input) + shortcut
        
        shortcut = input
        input = self.layerNorm2(input)
        input = self.feedForward(input)
        input = self.dropoutShortcut(input) + shortcut
        return input
    }
}
