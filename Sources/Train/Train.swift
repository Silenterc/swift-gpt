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

@main
struct Train {
    private static let dataDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // Train/
        .deletingLastPathComponent() // Sources/
        .deletingLastPathComponent() // swift-gpt/
        .appendingPathComponent("data/texts")
    
    private static let tokensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // Train/
        .deletingLastPathComponent() // Sources/
        .deletingLastPathComponent() // swift-gpt/
        .appendingPathComponent("tokens")
    
    private static let config = GPTConfig.gpt2Small
    
    private static let batchSize = 16
    private static let stride = config.contextLength // No overlap
    
    static func main() async {
        MLXRandom.seed(67)
        
        do {
            let fileReader = LocalFileReader()
            let tokenizer = try await getTokenizer()
            let writer = try TokenFileWriter(outputDirectory: tokensDir, maxTokensPerShard: 100_000_000)
            defer { try? writer.finish() }
            
            for file in try FileManager.default.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: nil) {
                let text = try fileReader.read(from: file)
                var tokens = tokenizer.encode(text: text).map(UInt32.init)
                // Add special End Of Sequence token after each text
                tokens.append(UInt32(tokenizer.eosTokenId!))
                
                try writer.write(tokens)
            }
            
            let loader = try DataLoader(
                dataset: TokenDataset(tokenDirectory: tokensDir),
                batchSize: batchSize,
                maxLength: config.contextLength,
                stride: stride
            )
            
            let model = GPTModel(config: config)
            
            // For now this only performs forward passes
            while let batch = try loader.nextBatch() {
                let logits = model(batch.inputIds)
                
                // inputIds:  [batch, tokens]
                // targetIds: [batch, tokens]
                // logits:    [batch, tokens, vocabSize]
                print("Input:  \(batch.inputIds.shape)")
                print("Target: \(batch.targetIds.shape)")
                print("Logits: \(logits.shape)")
            }
            
        } catch {
            print("Error: \(error)")
        }
    }
}

