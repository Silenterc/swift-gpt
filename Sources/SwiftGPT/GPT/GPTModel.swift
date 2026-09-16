//
//  GPTModel.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

import MLX
import MLXNN

public class GPTModel: Module, UnaryLayer {
    @ModuleInfo var tokenEmbedding: Embedding
    @ModuleInfo var positionalEmbedding: Embedding
    @ModuleInfo var dropout: Dropout
    @ModuleInfo var transformerBlocks: Sequential
    @ModuleInfo var outputNorm: LayerNorm
    @ModuleInfo var outProj: Linear // We dont use weight tying
    
    let contextLength: Int
    
    public init(config: GPTConfig) {
        self.contextLength = config.contextLength
        self.tokenEmbedding = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.embeddingDimension
        )
        self.positionalEmbedding = Embedding(
            embeddingCount: config.contextLength,
            dimensions: config.embeddingDimension
        )
        self.dropout = Dropout(p: config.dropoutRate)
        self.transformerBlocks = Sequential(
            layers: (0..<config.numLayers).map { _ in
                TransformerBlock(config: config)
            }
        )
        self.outputNorm = LayerNorm(embeddingDimension: config.embeddingDimension)
        self.outProj = Linear(
            config.embeddingDimension,
            config.vocabSize,
            bias: false
        )
    }
    
    /**
     Performs a forward pass through the GPT model

     - Parameter x: Token IDs with shape `[batch, tokens]`
     - Returns: Logits with shape `[batch, tokens, vocabSize]`
     */
    public func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        assert(x.ndim == 2) // We want [batch, tokens]
        let sequenceLength = x.dim(-1)
        assert(sequenceLength <= self.contextLength)
        
        let tokenEmbeddings = self.tokenEmbedding(x)
        let positionalEmbeddings = self.positionalEmbedding(MLXArray(0..<sequenceLength))
        var output = tokenEmbeddings + positionalEmbeddings
        output = self.dropout(output)
        output = self.transformerBlocks(output)
        output = self.outputNorm(output)
        output = self.outProj(output)
        return output
    }
}
