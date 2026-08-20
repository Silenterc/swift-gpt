//
//  Tokenizer.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 15.08.2026.
//
import Foundation
import MLXLMCommon

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol MyTokenizer: Sendable {
    func encode(_ text: String) -> [Int]
    func decode(_ tokenIds: [Int]) -> String
}

public struct SimpleTokenizer: MyTokenizer {
    private let vocab: [String: Int]
    private let inverseVocab: [Int: String]
    
    // Matches common separators
    // Regex<> is not Sendable so these cannot be stored properties...
    private static var separatorRegex: Regex<Substring> { /[,.:;?_!"()']|--|\s/ }
    // Matches separators which do NOT have a space before them
    private static var decodeSeparatorRegex: Regex<(Substring, Substring)> { /\s+([,.?!"()'])/ }
    public static let unknownToken = "<unk>"
    
    init(vocab: [String: Int]) {
        assert(vocab[Self.unknownToken] != nil)
        
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
        
        for match in text.matches(of: Self.separatorRegex) {
            parts.append(text[start..<match.range.lowerBound]) // Add the preceding token
            parts.append(text[match.range]) // Add the separator token
            start = match.range.upperBound // Update the index
        }
        parts.append(text[start...]) // Add the remaining token
        
        let ids = parts
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { self.vocab[$0] ?? self.vocab[Self.unknownToken]! } // Lookup the id or add <|unk|>
        
        return ids
    }
    
    public func decode(_ tokenIds: [Int]) -> String {
        let text = tokenIds.map { self.inverseVocab[$0] ?? Self.unknownToken }.joined(separator: " ")
        let strippedText = text.replacing(Self.decodeSeparatorRegex) { match in
            String(match.1)
        }
        return strippedText
    }
}

/**
 The functions below are mostly gonna be very dummy implementations because
 Im not gonna need those features, I just want the conformance to Tokenizer
 */
extension SimpleTokenizer : MLXLMCommon.Tokenizer {
    public func applyChatTemplate(messages: [[String : any Sendable]], tools: [[String : any Sendable]]?, additionalContext: [String : any Sendable]?) throws -> [Int] {
        []
    }
    
    public static let bosToken = "<bos>"
    public static let eosToken = "<eos>"
    
    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        decode(tokenIds)
    }
    
    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        encode(text)
    }
    
    public func convertTokenToId(_ token: String) -> Int? {
        vocab[token]
    }
    
    public func convertIdToToken(_ id: Int) -> String? {
        inverseVocab[id]
    }
    
    public var bosToken: String? {
        Self.bosToken
    }
    
    public var eosToken: String? {
        Self.eosToken
    }
    
    public var unknownToken: String? {
        Self.unknownToken
    }
    
    
}

