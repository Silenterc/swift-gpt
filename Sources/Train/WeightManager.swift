//
//  WeightManager.swift
//  swift-gpt
//
//  Created by Lukáš Zima on 02.10.2026.
//

import Foundation
import MLXNN
import MLX

/**
 Used for saving and loading weights to disk
 */
struct WeightManager {
    
    /**
    Saves the module weights to a SafeTensors file

    - Parameters:
        - module: The module whose weights should be saved
        - url: The destination URL. Must have the `.safetensors` file extension
    */
    static func save(module: Module, to url: URL) throws {
        let weights = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        try MLX.save(arrays: weights, url: url)
    }
    
    /**
    Loads weights from a SafeTensors file into the module

    - Parameters:
        - module: The module whose weights should be updated
        - url: The URL of the `.safetensors` file to load
    */
    static func load(module: Module, from url: URL) throws {
        let weights = try MLX.loadArrays(url: url)
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    }
}
