//
//  FileHandlingTests.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 18.08.2026.
//

import Testing
import Foundation
@testable import SwiftGPT

func makeTestDir() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent("tokens-\(UUID().uuidString)")
}

@Test("TokenFileWriter init creates a dir")
func tokenFileWriterInitCreatesDir() async throws {
    let testDirURL = makeTestDir()
    let _ = try TokenFileWriter(outputDirectory: testDirURL, maxTokensPerShard: 10)
    #expect(FileManager.default.fileExists(atPath: testDirURL.path))
    try? FileManager.default.removeItem(at: testDirURL)
}

@Test("TokenFileWriter writes tokens to one file")
func tokenFileWriterWritesToFile() async throws {
    let testDirURL = makeTestDir()
    let writer = try TokenFileWriter(outputDirectory: testDirURL, maxTokensPerShard: 100)
    let tokens1: [UInt32] = [1, 2, 3, 4, 5]
    let tokens2: [UInt32] = [6, 7, 8, 9, 10]
    try writer.write(tokens1)
    try writer.write(tokens2)
    
    let data = try Data(contentsOf: testDirURL.appendingPathComponent("tokens-00000.bin"))
    let out = data.withUnsafeBytes { ptr in
        Array(ptr.bindMemory(to: UInt32.self))
    }
    #expect(out == tokens1 + tokens2)
    
    try? FileManager.default.removeItem(at: testDirURL)
}

@Test("TokenFileWriter writes tokens to sharded files")
func tokenFileWriterWritesToFiles() async throws {
    let testDirURL = makeTestDir()
    let writer = try TokenFileWriter(outputDirectory: testDirURL, maxTokensPerShard: 5)
    let tokens1: [UInt32] = [1, 2, 3, 4, 5]
    let tokens2: [UInt32] = [6, 7, 8, 9, 10]
    try writer.write(tokens1)
    try writer.write(tokens2)
    
    let data1 = try Data(contentsOf: testDirURL.appendingPathComponent("tokens-00000.bin"))
    let data2 = try Data(contentsOf: testDirURL.appendingPathComponent("tokens-00001.bin"))
    
    let out1 = data1.withUnsafeBytes { ptr in
        Array(ptr.bindMemory(to: UInt32.self))
    }
    let out2 = data2.withUnsafeBytes { ptr in
        Array(ptr.bindMemory(to: UInt32.self))
    }
    #expect(out1 == tokens1)
    #expect(out2 == tokens2)
    
    try? FileManager.default.removeItem(at: testDirURL)
}

@Test("TokenDataset init fails for non-existing dir")
func tokenDatasetInitFailsForNonExistingDir() async throws {
    await #expect(processExitsWith: .failure) {
        let testDirURL = makeTestDir()
        try? FileManager.default.removeItem(at: testDirURL)
        let _ = try TokenDataset(tokenDirectory: testDirURL)
    }
}

@Test("TokenFileWriter + TokenDataset work 1 file")
func tokenFileWriterTokenDatasetWork1file() async throws {
    let testDirURL = makeTestDir()
    let writer = try TokenFileWriter(outputDirectory: testDirURL, maxTokensPerShard: 10)
    let tokens: [UInt32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    try writer.write(tokens)
    
    let ds = try TokenDataset(tokenDirectory: testDirURL)
    let gotTokens = try ds.getTokens(at: 0, count: 10)
    #expect(gotTokens == tokens)
    try? FileManager.default.removeItem(at: testDirURL)
}

@Test("TokenFileWriter + TokenDataset work sharded files")
func tokenFileWriterTokenDatasetWorkShardedFiles() async throws {
    let testDirURL = makeTestDir()
    let writer = try TokenFileWriter(outputDirectory: testDirURL, maxTokensPerShard: 2)
    let tokens: [UInt32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    try writer.write(tokens)
    
    let ds = try TokenDataset(tokenDirectory: testDirURL)
    let gotTokens = try ds.getTokens(at: 0, count: 10)
    #expect(gotTokens == tokens)
    try? FileManager.default.removeItem(at: testDirURL)
}

@Test("TokenFileWriter + TokenDataset work sharded files 2")
func tokenFileWriterTokenDatasetWorkShardedFiles2() async throws {
    let testDirURL = makeTestDir()
    let writer = try TokenFileWriter(outputDirectory: testDirURL, maxTokensPerShard: 2)
    let tokens1: [UInt32] = [0, 1]
    let tokens2: [UInt32] = [2, 3]
    let tokens3: [UInt32] = [4, 5]
    let tokens4: [UInt32] = [6, 7]
    let tokens5: [UInt32] = [8, 9]
    try writer.write(tokens1)
    try writer.write(tokens2)
    try writer.write(tokens3)
    try writer.write(tokens4)
    try writer.write(tokens5)
    
    let ds = try TokenDataset(tokenDirectory: testDirURL)
    let gotTokens = try ds.getTokens(at: 0, count: 5)
    #expect(gotTokens == [0, 1, 2, 3, 4])
    let gotTokens2 = try ds.getTokens(at: 5, count: 5)
    #expect(gotTokens2 == [5, 6, 7, 8, 9])
    let gotTokens3 = try ds.getTokens(at: 0, count: 10000)
    #expect(gotTokens3.count == 10)
    let gotTokens4 = try ds.getTokens(at: 3, count: 5)
    #expect(gotTokens4 == [3, 4, 5, 6, 7])
    let noTokens = try ds.getTokens(at: 10, count: 1)
    #expect(noTokens.isEmpty)
    try? FileManager.default.removeItem(at: testDirURL)
}
