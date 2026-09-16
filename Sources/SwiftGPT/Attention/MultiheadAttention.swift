//
//  MultiheadAttention.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 04.09.2026.
//

import MLXNN
import MLX

public class MultiheadAttention: Module, UnaryLayer {
    @ModuleInfo var wQuery: Linear
    @ModuleInfo var wKey: Linear
    @ModuleInfo var wValue: Linear
    
    let dropout: Dropout
    let causalMask: MLXArray
    
    let outputDimensions: Int
    let numHeads: Int
    let headDim: Int
    
    @ModuleInfo var outProj: Linear
    
    /**
     Initializes the `MultiheadAttention`

     - Parameters:
       - inputDimensions: The input embedding dimensions
       - outputDimensions: The desired output dimensions
       - contextLength: The maximum context length - number of tokens
       - numHeads: The number of attention heads
       - dropout: The dropout probability
       - bias: Whether the linear layers should use a bias
     */
    public init(inputDimensions: Int, outputDimensions: Int, contextLength: Int,
                numHeads: Int, dropout: Float = 0.0, bias: Bool = false) {
        // The output dim must be divisible by numHeads
        assert(numHeads > 0 && outputDimensions % numHeads == 0)
        
        self.wQuery = Linear(inputDimensions, outputDimensions, bias: bias)
        self.wKey = Linear(inputDimensions, outputDimensions, bias: bias)
        self.wValue = Linear(inputDimensions, outputDimensions, bias: bias)
        
        self.dropout = Dropout(p: dropout)
        let mask = MLXArray.tri(contextLength)
        // causalMask contains -INF in the upper triangle and 0s on the diagonal and below it
        self.causalMask = MLX.where(mask .== 1, 0, -Float.infinity)
        
        self.outputDimensions = outputDimensions
        self.numHeads = numHeads
        self.headDim = outputDimensions / numHeads
        
        self.outProj = Linear(outputDimensions, outputDimensions)
        
        super.init()
        self.freeze(recursive: false, keys: ["causalMask"]) // Mask is non-trainable
    }
    
    public func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        assert(x.ndim >= 2  && x.ndim <= 3)
        
        let is3d = x.ndim == 3 // We want to work with batches
        let input = is3d ? x : x.expandedDimensions(axis: 0)
        let (batch, numTokens, inputDim) = input.shape3
        
        let queries = self.wQuery(input)
            .reshaped(batch, numTokens, self.numHeads, self.headDim) // Split among heads
            .swappedAxes(1, 2) // Move the head dim before the token dim so matmul operates per head
        let keys = self.wKey(input)
            .reshaped(batch, numTokens, self.numHeads, self.headDim)
            .swappedAxes(1, 2)
        let values = self.wValue(input)
            .reshaped(batch, numTokens, self.numHeads, self.headDim)
            .swappedAxes(1, 2)
        
        let attentionScores = queries.matmul(keys.swappedAxes(2, 3))
        let attentionWeights = self.getAttentionWeights(
            attentionScores: attentionScores,
            numTokens: numTokens
        )
        
        var contextVectors = attentionWeights.matmul(values)
            .swappedAxes(1, 2) // Bring back the shape
            .reshaped(batch, numTokens, self.outputDimensions) // Merge the heads
        if (!is3d) {
            // Remove the artificial batch dimension added for 2d input
            contextVectors = contextVectors.squeezed(axis: 0)
        }
        
        return self.outProj(contextVectors)
    }
    
    /**
     Performs scaling, causal masking, softmax and dropout
     And returns the attention weights
     */
    private func getAttentionWeights(attentionScores: MLXArray, numTokens: Int) -> MLXArray {
        let scaledAttentionScores = attentionScores / Float(self.headDim).squareRoot()
        let mask = self.causalMask[..<numTokens, ..<numTokens]
        let maskedAttentionScores = scaledAttentionScores + mask
        let attentionWeights = softmax(maskedAttentionScores, axis: -1)
        let droppedAttentionWeights = self.dropout(attentionWeights)
        return droppedAttentionWeights
    }
    
}
