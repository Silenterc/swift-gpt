//
//  TokenizerTests.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 15.08.2026.
//

import Testing
@testable import SwiftGPT

let vocab = [
    "Hello": 0,
    ",": 1,
    "world": 2,
    "!": 3,
    SimpleTokenizer.unknownToken: 4
]

@Test("Test encode 1")
func encode1() async throws {
    let tokenizer = SimpleTokenizer(vocab: vocab)
    let ids = tokenizer.encode("Hello, world!")
    
    #expect(ids == [0, 1, 2, 3])
}

@Test("Test encode 2")
func encode2() async throws {
    let tokenizer = SimpleTokenizer(vocab: vocab)
    let ids = tokenizer.encode("Hello, Jeff.")
    
    #expect(ids == [0, 1, 4, 4])
}

@Test("Test encode Empty")
func encode3() async throws {
    let tokenizer = SimpleTokenizer(vocab: vocab)
    let ids = tokenizer.encode(" ")
    
    #expect(ids == [])
}

@Test("Test decode 1")
func decode1() async throws {
    let tokenizer = SimpleTokenizer(vocab: vocab)
    let text = tokenizer.decode([0, 1, 2, 3])
    
    #expect(text == "Hello, world!")
}

@Test("Test decode 2")
func decode2() async throws {
    let tokenizer = SimpleTokenizer(vocab: vocab)
    let text = tokenizer.decode([0, 1, 4, 4])
    
    #expect(text == "Hello, \(SimpleTokenizer.unknownToken) \(SimpleTokenizer.unknownToken)")
}

@Test("Test encode + decode")
func encodeDecode() async throws {
    let tokenizer = SimpleTokenizer(vocab: vocab)
    let ids = tokenizer.encode("Hello, world!")
    let text = tokenizer.decode(ids)
    
    #expect(text == "Hello, world!")
}

@Test("Test constructor fails due to missing unk token")
func shouldFail() async throws {
    await #expect(processExitsWith: .failure) {
        var badVocab = vocab
        badVocab.removeValue(forKey: SimpleTokenizer.unknownToken)
        let _ = SimpleTokenizer(vocab: badVocab)
    }
}


