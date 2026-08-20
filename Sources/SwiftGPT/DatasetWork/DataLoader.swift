//
//  DataLoader.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 19.08.2026.
//

import Foundation
import MLX

/// Loads a pair of `(inputIds, targetIds)` from the given `dataset: TokenDataset`
/// Uses sliding-window sampling to generate inputs and targets
public class DataLoader {
    private let dataset: TokenDataset
    private let batchSize: Int
    private let maxLength: Int
    private let stride: Int
    
    private var currentIndex = 0
    
    /// Initializes the `DataLoader`
    /// - parameter dataset: The dataset to load the tokens from
    /// - parameter batchSize: The number of sequences in each batch
    /// - parameter maxLength: The maximum number of tokens in each sequence
    /// - parameter stride: The step size between the sequences
    public init(dataset: TokenDataset, batchSize: Int, maxLength: Int, stride: Int) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.maxLength = maxLength
        self.stride = stride
    }
    
    /// Loads the next batch of input and target token IDs
    /// Returns `nil` when there are none left
    /// Both output tensors have the shape `[completedBatches, maxlength]`
    public func nextBatch() throws -> (inputIds: MLXArray, targetIds: MLXArray)? {
        var inputIdsRet: [UInt32] = []
        var targetsIdsRet: [UInt32] = []
        
        inputIdsRet.reserveCapacity(batchSize * maxLength)
        targetsIdsRet.reserveCapacity(batchSize * maxLength)
        
        var completedBatches = 0
        
        for _ in 0 ..< batchSize {
            let ids = try dataset.getTokens(at: currentIndex, count: maxLength + 1)
            // Are we at the end?
            guard ids.count == maxLength + 1 else {
                break
            }
            
            inputIdsRet.append(contentsOf: ids.dropLast())
            targetsIdsRet.append(contentsOf: ids.dropFirst())
            
            completedBatches += 1
            currentIndex += stride
        }
        
        guard completedBatches > 0 else {
            return nil
        }
        
        return (
            MLXArray(inputIdsRet, [completedBatches, maxLength]),
            MLXArray(targetsIdsRet, [completedBatches, maxLength])
        )
    }
}
