//
//  FeedForward.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.09.2026.
//

import MLX
import MLXNN

public class FeedForward: Module, UnaryLayer {
    @ModuleInfo var layers: Sequential
    
    /**
     Initializes the `FeedForward` layer

     - Parameters:
        - embeddingDimension: The input and output embedding dimensions
        - hiddenMultiplier: The multiplier used to determine the hidden layer dimensions
    */
    public init(embeddingDimension: Int, hiddenMultiplier: Int = 4) {
        self.layers = Sequential(layers: [
            Linear(embeddingDimension, hiddenMultiplier * embeddingDimension),
            GELU(),
            Linear(hiddenMultiplier * embeddingDimension, embeddingDimension)
        ])
    }
    
    public func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        layers(x)
    }
}
