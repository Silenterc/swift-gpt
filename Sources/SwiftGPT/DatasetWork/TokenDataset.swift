//
//  TokenDataset.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 18.08.2026.
//

import Foundation

private struct TokenShard {
    let url: URL
    let tokenCount: Int
    let startTokenIndex: Int
    
    init(url: URL, tokenCount: Int, startTokenIndex: Int) {
        self.url = url
        self.tokenCount = tokenCount
        self.startTokenIndex = startTokenIndex
    }
}

/// Reads and provides tokens saved by `TokenFileWriter` in the `tokenDirectory`
public class TokenDataset {
    private let tokenDirectory: URL
    private let availableTokens: Int
    private var shards: [TokenShard]
    
    private var currentShardIndex: Int?
    private var currentShardData: Data?
    
    public init(tokenDirectory: URL) throws {
        assert(FileManager.default.fileExists(atPath: tokenDirectory.path))
        self.tokenDirectory = tokenDirectory
        
        var availableTokens = 0
        var shards: [TokenShard] = []
        
        // Iterate over the token files and calculate the amount of tokens in them
        let urls = try FileManager.default.contentsOfDirectory(
            at: tokenDirectory,
            includingPropertiesForKeys: [.fileSizeKey]
        )
            .filter { $0.lastPathComponent.hasPrefix("tokens-") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent } // Sort by their numbering
        
        for url in urls {
            guard let fileSize = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize else {
                self.availableTokens = 0
                self.shards = []
                return
            }
            let tokenCount = Int(fileSize / 4) // UInt32 is 4 bytes
            
            shards.append(TokenShard(url: url, tokenCount: tokenCount, startTokenIndex: availableTokens))
            availableTokens += tokenCount
        }
        
        self.availableTokens = availableTokens
        self.shards = shards
    }
    
    /// Get `count` of tokens (their ids) at a given `offset`
    /// Works across the whole sharded `tokenDirectory`
    public func getTokens(at offset: Int, count: Int) throws -> [UInt32] {
        assert(offset >= 0 && offset <= availableTokens)
        assert(count >= 0)
        let readCount = min(count, availableTokens - offset)
        var remainingCount = readCount
        var currentOffset = offset
        
        var tokens: [UInt32] = []
        tokens.reserveCapacity(readCount) 
        while (tokens.count < readCount) {
            guard let shardIndex = getShardIndex(containing: currentOffset) else {
                break
            }
            try mapShard(at: shardIndex)
            
            // Calculate and clamp how many tokens we can read in the current shard
            let shard = shards[shardIndex]
            let offsetInShard = currentOffset - shard.startTokenIndex
            let remainingInShard = shard.tokenCount - offsetInShard
            let tokensToRead = min(remainingCount, remainingInShard)
            
            // Read those tokens
            self.currentShardData!.withUnsafeBytes { ptr in
                let tokenBuffer = ptr.bindMemory(to: UInt32.self)
                let range = offsetInShard ..< offsetInShard + tokensToRead
                let chunk = tokenBuffer[range].map { UInt32(littleEndian: $0) }
                tokens += chunk
            }
            
            // Update and possibly move on to the next shard
            currentOffset += tokensToRead
            remainingCount -= tokensToRead
        }
        return tokens
           
    }
    
    private func getShardIndex(containing offset: Int) -> Int? {
        // Try the cached shard, will work for sequential reading
        if let currentShardIndex {
            let currentShard = shards[currentShardIndex]
            if offset >= currentShard.startTokenIndex && offset < currentShard.startTokenIndex + currentShard.tokenCount {
                return currentShardIndex
            }
            
            // Also try the following shard
            let nextShardIndex = currentShardIndex + 1
            if nextShardIndex < shards.count {
                let nextShard = shards[nextShardIndex]
                if offset >= nextShard.startTokenIndex && offset < nextShard.startTokenIndex + nextShard.tokenCount {
                    return nextShardIndex
                }
            }
        }
        
        // Fallback, could be done by bin search but probs not necessary for now
        return shards.firstIndex {
            offset >= $0.startTokenIndex &&
            offset < $0.startTokenIndex + $0.tokenCount
        }
    }
    
    /// Memory maps the shard
    private func mapShard(at index: Int) throws {
        currentShardData = try Data(contentsOf: shards[index].url, options: .alwaysMapped)
        currentShardIndex = index
    }
}
