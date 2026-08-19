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

let simpleVocabulary: [String: Int] = [
    SimpleTokenizer.unknownToken: 0,
    "the": 1,
    "and": 2,
    "a": 3,
    "I": 4,
    "to": 5,
    "of": 6,
    "was": 7,
    "in": 8,
    "he": 9,
    "that": 10,
    "you": 11,
    "it": 12,
    "his": 13,
    "with": 14,
    "had": 15,
    
    "Gatsby": 16,
    "Daisy": 17,
    "Nick": 18,
    
    ".": 19,
    ",": 20,
    "!": 21,
    "?": 22
]

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

@Test("Test the whole flow on The Great Gatsby book")
func testTheGreatGatsby() async throws {
    let testDirURL = makeTestDir()
    let gatsbyURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // SwiftGPTTests
        .deletingLastPathComponent() // Tests
        .deletingLastPathComponent() // swift-gpt
        .appendingPathComponent("data/The-Great-Gatsby.txt")
    
    let fileReader = LocalFileReader()
    let text = try fileReader.read(from: gatsbyURL)
    let writer = try TokenFileWriter(outputDirectory: testDirURL, maxTokensPerShard: 10_000)
    let simpleTokenizer = SimpleTokenizer(vocab: simpleVocabulary)
    
    let tokenized = simpleTokenizer.encode(text).map(UInt32.init)
    try writer.write(tokenized)
    
    let ds = try TokenDataset(tokenDirectory: testDirURL)
    let allTokens = try ds.getTokens(at: 0, count: tokenized.count)
    
    #expect(allTokens.count == tokenized.count)
    try? FileManager.default.removeItem(at: testDirURL)
}

@Test("Test the DataLoader 1")
func testDataLoader1() async throws {
    let testDirURL = makeTestDir()
    let writer = try TokenFileWriter(outputDirectory: testDirURL, maxTokensPerShard: 10_000)
    let tokens: [UInt32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    try writer.write(tokens)
    let ds = try TokenDataset(tokenDirectory: testDirURL)
    
    
    let loader = DataLoader(dataset: ds, batchSize: 5, maxLength: 2, stride: 1)
    let (inputs, targets) = try loader.nextBatch()!
    
    #expect(inputs.shape == [5, 2] && targets.shape == [5, 2])
    
    #expect(
        inputs[0, 0].item() == 0 && inputs[0, 1].item() == 1 &&
        targets[0, 0].item() == 1 && targets[0, 1].item() == 2
    )
    
    #expect(
        inputs[4, 0].item() == 4 && inputs[4, 1].item() == 5 &&
        targets[4, 0].item() == 5 && targets[4, 1].item() == 6
    )
    
    try? FileManager.default.removeItem(at: testDirURL)
}

@Test("Test the DataLoader with stride")
func testDataLoaderStride() async throws {
    let testDirURL = makeTestDir()
    let writer = try TokenFileWriter(outputDirectory: testDirURL, maxTokensPerShard: 10_000)
    let tokens: [UInt32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    try writer.write(tokens)
    let ds = try TokenDataset(tokenDirectory: testDirURL)
    
    
    let loader = DataLoader(dataset: ds, batchSize: 5, maxLength: 2, stride: 2)
    let (inputs, targets) = try loader.nextBatch()!
    
    #expect(inputs.shape == [4, 2] && targets.shape == [4, 2])
    
    #expect(
        inputs[0, 0].item() == 0 && inputs[0, 1].item() == 1 &&
        targets[0, 0].item() == 1 && targets[0, 1].item() == 2
    )
    
    #expect(
        inputs[1, 0].item() == 2 && inputs[1, 1].item() == 3 &&
        targets[1, 0].item() == 3 && targets[1, 1].item() == 4
    )
    
    #expect(
        inputs[3, 0].item() == 6 && inputs[3, 1].item() == 7 &&
        targets[3, 0].item() == 7 && targets[3, 1].item() == 8
    )
    
    try? FileManager.default.removeItem(at: testDirURL)
}
