//
//  DatasetReader.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 16.08.2026.
//

import Foundation

/// Reads a file to memory as String
public struct LocalFileReader{
    public init() {}
    
    public func read(from url: URL) throws -> String {
        try String(contentsOf: url, encoding: .utf8)
    }
}
