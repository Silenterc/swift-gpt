//
//  PrepareDataset.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 20.09.2026.
//
import Foundation
import MLXLMCommon
import SwiftGPT

@main
struct PrepareDataset {
    private static let seed: UInt64 = 67
    private static let trainFraction: Double = 0.99

    private static let baseDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // Train/
        .deletingLastPathComponent() // Sources/
        .deletingLastPathComponent() // swift-gpt/

    private static let dataDir = Self.baseDir
        .appendingPathComponent("data/project-gutenberg/books")

    private static let trainTokensDir = Self.baseDir
        .appendingPathComponent("tokens/train")

    private static let valTokensDir = Self.baseDir
        .appendingPathComponent("tokens/val")

    static func main() async {
        do {
            var allBooks = try FileManager.default
                .contentsOfDirectory(at: dataDir, includingPropertiesForKeys: nil)
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            var rng = SeededRandomNumberGenerator(seed: Self.seed)
            // Shuffle the books so they are truly randomly distributed
            allBooks.shuffle(using: &rng)

            // Split the data between training (99%) and validation (1%)
            let splitIndex = Int(Double(allBooks.count) * Self.trainFraction)
            let trainBooks = allBooks[..<splitIndex]
            let valBooks = allBooks[splitIndex...]

            // Setup reader and writers
            let fileReader = LocalFileReader()
            let tokenizer = try await getTokenizer()
            let trainWriter = try TokenFileWriter(outputDirectory: trainTokensDir, maxTokensPerShard: 100_000_000)
            let valWriter = try TokenFileWriter(outputDirectory: valTokensDir, maxTokensPerShard: 100_000_000)
            defer { try? trainWriter.finish(); try? valWriter.finish() }

            try tokenizeAndWrite(
                files: trainBooks,
                splitName: "train",
                fileReader: fileReader,
                tokenizer: tokenizer,
                writer: trainWriter
            )
            try tokenizeAndWrite(
                files: valBooks,
                splitName: "val",
                fileReader: fileReader,
                tokenizer: tokenizer,
                writer: valWriter
            )

        } catch {
            print("Error: \(error)")
        }
    }

    private static func tokenizeAndWrite(
        files: ArraySlice<URL>,
        splitName: String,
        fileReader: LocalFileReader,
        tokenizer: any MLXLMCommon.Tokenizer,
        writer: TokenFileWriter
    ) throws {
        print("Tokenizing \(splitName) split: \(files.count) books")

        for (index, file) in files.enumerated() {
            let text = try fileReader.read(from: file)
            var tokens = tokenizer.encode(text: text).map(UInt32.init)
            // Add special End Of Sequence token after each text
            tokens.append(UInt32(tokenizer.eosTokenId!))

            try writer.write(tokens)

            let completedCount = index + 1
            if completedCount.isMultiple(of: 100) || completedCount == files.count {
                print("[\(splitName)] \(completedCount)/\(files.count) books tokenized")
            }
        }
    }
}
