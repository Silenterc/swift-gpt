//
//  Tokenizer.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 15.08.2026.
//
import Foundation

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ tokenIds: [Int]) -> String
}

public struct SimpleTokenizer: Tokenizer {
    private let vocab: [String: Int]
    private let inverseVocab: [Int: String]
    
    // Matches common separators
    private let separatorRegex = /([,.:;?_!"()']|--|\s)/
    // Matches separators which do NOT have a space before them
    private let decodeSeparatorRegex = /\s+([,.?!"()'])/
    public static let unknownToken = "<|unk|>"
    
    init(vocab: [String: Int]) {
        assert(vocab[SimpleTokenizer.unknownToken] != nil)
        
        self.vocab = vocab
        var invVocab: [Int: String] = [:]
        for (k, v) in vocab {
            invVocab[v] = k
        }
        self.inverseVocab = invVocab
    }
    
    public func encode(_ text: String) -> [Int] {
        var parts: [Substring] = []
        var start = text.startIndex
        
        for match in text.matches(of: self.separatorRegex) {
            parts.append(text[start..<match.range.lowerBound]) // Add the preceding token
            parts.append(text[match.range]) // Add the separator token
            start = match.range.upperBound // Update the index
        }
        parts.append(text[start...]) // Add the remaining token
        
        let ids = parts
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { self.vocab[$0] ?? self.vocab[SimpleTokenizer.unknownToken]! } // Lookup the id or add <|unk|>
        
        return ids
    }
    
    public func decode(_ tokenIds: [Int]) -> String {
        let text = tokenIds.map { self.inverseVocab[$0] ?? SimpleTokenizer.unknownToken }.joined(separator: " ")
        let strippedText = text.replacing(self.decodeSeparatorRegex) { match in
            String(match.1)
        }
        return strippedText
    }
    
    
}

