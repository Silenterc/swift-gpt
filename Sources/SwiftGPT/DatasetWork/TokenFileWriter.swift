//
//  TokenFileWriter.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 17.08.2026.
//

import Foundation

/// Writes tokens to disk as continuous sharded binary files
public class TokenFileWriter {
    private let outputDirectory: URL
    private let maxTokensPerShard: Int
    
    private var shardIndex: Int
    private var tokensInShard = 0
    private var fileHandle: FileHandle?
    
    public init(outputDirectory: URL, maxTokensPerShard: Int, shardIndex: Int = 0) throws {
        self.outputDirectory = outputDirectory
        self.maxTokensPerShard = maxTokensPerShard
        self.shardIndex = shardIndex
        // Create the dir if it doesnt exist for some reason
        if !FileManager.default.fileExists(atPath: outputDirectory.path) {
            try FileManager.default.createDirectory(
                at: outputDirectory,
                withIntermediateDirectories: true
            )
        }
        try createNextShard()
    }
    
    /// Writes the tokens array into managed sharded files
    /// Creates new continuous shards when needed during one session
    public func write(_ tokens: [UInt32]) throws {
        var offset = 0
        
        while offset < tokens.count {
            let capacity = self.maxTokensPerShard - self.tokensInShard
            let count = min(capacity, tokens.count - offset)
            
            let chunk = tokens[offset ..< offset + count].map(\.littleEndian)
            let chunkData = chunk.withUnsafeBytes { Data($0) }
            try self.fileHandle?.write(contentsOf: chunkData)
            
            offset += count
            self.tokensInShard += count
            
            if self.tokensInShard >= self.maxTokensPerShard {
                try createNextShard()
            }
        }
    }
    
    public func finish() throws {
        try self.fileHandle?.close()
        self.fileHandle = nil
    }
    
    private func createNextShard() throws {
        try self.fileHandle?.close()
        
        let fileName = String(format: "tokens-%05d.bin", shardIndex)
        let fileURL = outputDirectory.appendingPathComponent(fileName)
        
        FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        self.fileHandle = try FileHandle(forWritingTo: fileURL)
        
        shardIndex += 1
        tokensInShard = 0
    }
}
