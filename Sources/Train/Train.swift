//
//  Train.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 11.08.2026.
//

import Foundation
import SwiftGPT
import MLX
import MLXNN
import MLXOptimizers

@main
struct Train {
    private static let baseDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // Train/
        .deletingLastPathComponent() // Sources/
        .deletingLastPathComponent() // swift-gpt/

    private static let trainTokensDir = Self.baseDir
        .appendingPathComponent("tokens/train")

    private static let valTokensDir = Self.baseDir
        .appendingPathComponent("tokens/val")
    
    private static let saveWeightsDir = Self.baseDir
        .appendingPathComponent("weights")
    private static let config = GPTConfig.gpt2Small
    
    private static let batchSize = 8
    private static let stride = config.contextLength // No overlap
    
    static func main() async {
        MLXRandom.seed(67)
        
        do {
            let trainLoader = try DataLoader(
                dataset: TokenDataset(tokenDirectory: trainTokensDir),
                batchSize: batchSize,
                maxLength: config.contextLength,
                stride: stride
            )
            
            let valLoader = try DataLoader(
                dataset: TokenDataset(tokenDirectory: valTokensDir),
                batchSize: batchSize,
                maxLength: config.contextLength,
                stride: stride
            )
            
            let model = GPTModel(config: config)
            let optimizer = AdamW(learningRate: 0.0004, weightDecay: 0.1, biasCorrection: true)
            let lossAndGrad = valueAndGrad(model: model, loss)
            
            var globalStep = 0
            let printLossFrequency = 50
            let evalFrequency = 1000
            let saveFrequency = 10_000
            
            model.train()
            
            while let batch = try trainLoader.nextBatch() {
                globalStep += 1
                
                let (trainLoss, grads) = lossAndGrad(model, batch.inputIds, batch.targetIds)
                
                optimizer.update(model: model, gradients: grads)
                
                eval(model, optimizer, trainLoss) // MLX Swift is lazy!
                
                if (globalStep % evalFrequency == 0) {
                    try evaluateAndPrint(model, trainLoss, valLoader, globalStep)
                } else if (globalStep % printLossFrequency == 0) {
                    printLoss(trainLoss: trainLoss, globalStep: globalStep)
                }
                
                if (globalStep % saveFrequency == 0) {
                    try CheckpointManager.save(
                        module: model, dataLoader: trainLoader,
                        globalStep: globalStep, to: saveWeightsDir
                    ) // We overwrite it, thats fine
                }
            }
            try CheckpointManager.save(
                module: model, dataLoader: trainLoader,
                globalStep: globalStep, to: saveWeightsDir
            ) // Save the final model
            
        } catch {
            print("Error: \(error)")
        }
    }
    
    private static func loss(model: GPTModel, inputs: MLXArray, targets: MLXArray) -> MLXArray {
        let logits = model(inputs)
        return crossEntropy(logits: logits, targets: targets, reduction: .mean)
    }
    
    private static func printLoss(trainLoss: MLXArray, globalStep: Int) {
        print("----- Step \(globalStep) -----")
        print("Training Loss: \(trainLoss.item(Float.self))")
    }
    
    private static func evaluateAndPrint(
        _ model: GPTModel,
        _ trainLoss: MLXArray,
        _ valLoader: DataLoader,
        _ globalStep: Int
    ) throws {
        model.train(false)
        defer { model.train() } // Set to training mode after this function finishes
        valLoader.reset()
        
        let validationBatches = 20
        var validationLoss = 0.0
        var batchCount = 0
        
        while batchCount < validationBatches,
              let batch = try valLoader.nextBatch() {
            let valLoss = loss(model: model, inputs: batch.inputIds, targets: batch.targetIds)
            validationLoss += Double(valLoss.item(Float.self)) // No need for eval(), item() runs it
            batchCount += 1
        }
        let averageValLoss = validationLoss / Double(batchCount)
        print("----- Step \(globalStep) Evaluation -----")
        print("Training Loss: \(trainLoss.item(Float.self))")
        print("Validation Loss: \(averageValLoss)")
    }
}

