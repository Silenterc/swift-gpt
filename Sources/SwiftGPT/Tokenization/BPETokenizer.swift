//
//  BPETokenizer.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 20.08.2026.
//

import MLXHuggingFace
import Tokenizers
import MLXLMCommon

/// Fetches a Byte-Pair Encoding Tokenizer used in GPT2
public func getTokenizer() async throws -> any MLXLMCommon.Tokenizer {
    let hfTokenizer = try await AutoTokenizer.from(pretrained: "openai-community/gpt2")
    let tokenizer = #adaptHuggingFaceTokenizer(hfTokenizer)
    return tokenizer
}
