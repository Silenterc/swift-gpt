//
//  Train.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 11.08.2026.
//

import Foundation
import SwiftGPT
import MLX

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
    
    private static let vocabSize = 50_257 // gpt2 BPE tokenizer vocab size
    private static let batchSize = 16
    private static let maxLength = 1024
    private static let stride = 512
    
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
                maxLength: maxLength,
                stride: stride
            )
            
        } catch {
            print("Error: \(error)")
        }
    }
}

