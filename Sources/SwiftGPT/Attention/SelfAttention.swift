//
//  SelfAttention.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 02.09.2026.
//

import MLXNN
import MLX

/**
 A simple self-attention module, using MLXArrays. It does not implement masking etc
 */
public class SelfAttention: Module, UnaryLayer {
    var wQuery: MLXArray
    var wKey: MLXArray
    var wValue: MLXArray
    
    public init(inputDimensions: Int, outputDimensions: Int) {
        self.wQuery = MLXRandom.uniform(low: 0, high: 1, [inputDimensions, outputDimensions])
        self.wKey = MLXRandom.uniform(low: 0, high: 1, [inputDimensions, outputDimensions])
        self.wValue = MLXRandom.uniform(low: 0, high: 1, [inputDimensions, outputDimensions])
    }
    
    /**
     Performs an attention pass
     - parameter x: Needs to be a 2D array of shape `[numTokens, inputDimensions]`
     */
    public func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        assert(x.ndim == 2)
        
        let queries = x.matmul(self.wQuery)
        let keys = x.matmul(self.wKey)
        let values = x.matmul(self.wValue)
        
        let attentionScores = queries.matmul(keys.T)
        let scaledAttentionScores = attentionScores / Float(keys.dim(-1)).squareRoot()
        let attentionWeights = softmax(scaledAttentionScores, axis: -1)
        
        let contextVectors = attentionWeights.matmul(values)
        return contextVectors
    }
}
