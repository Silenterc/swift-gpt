//
//  LayerNorm.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

import MLX
import MLXNN

public class LayerNorm: Module, UnaryLayer {
    @ParameterInfo var scale: MLXArray
    @ParameterInfo var shift: MLXArray
    
    let eps: Float
    
    public init(embeddingDimension: Int, eps: Float = 1e-5) {
        self.scale = MLXArray.ones([embeddingDimension])
        self.shift = MLXArray.zeros([embeddingDimension])
        self.eps = eps
    }
    
    /**
     Performs Layer Normalization
     */
    public func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let normalizedX = (x - mean) / MLX.sqrt(variance + self.eps)
        return self.scale * normalizedX + self.shift
    }
}
