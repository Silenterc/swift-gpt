//
//  CheckpointManager.swift
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
public struct CheckpointManager {
    
    struct State: Codable {
        let globalStep: Int
        let currentLoaderIndex: Int
    }
    
    /**
     Saves the module weights to a SafeTensors file
     
     - Parameters:
        - module: The module whose weights should be saved
        - dataLoader: The training data loader whose current position should be saved
        - globalStep: The current training step
        - directory: The directory in which the checkpoint should be saved
    */
    public static func save(module: Module, dataLoader: DataLoader, globalStep: Int, to directory: URL) throws {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let weightsURL = directory.appendingPathComponent("weights.safetensors")
        let stateURL = directory.appendingPathComponent("state.json")
        
        let weights = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        
        try MLX.save(arrays: weights,url: weightsURL)
        
        let state = State(globalStep: globalStep, currentLoaderIndex: dataLoader.getCurrentIndex())
        
        let data = try JSONEncoder().encode(state)
        try data.write(to: stateURL)
    }
    
    /**
      Loads the model weights and restores the training state.
     
      - Parameters:
        - module: The module whose weights should be loaded
        - dataLoader: The training data loader whose position should be restored
        - directory: The directory containing the checkpoint
        - Returns: The restored global training step
     */
    @discardableResult
    public static func load(module: Module, dataLoader: DataLoader, from directory: URL) throws -> Int {
        let weightsURL = directory.appendingPathComponent("weights.safetensors")
        let stateURL = directory.appendingPathComponent("state.json")
        
        let weights = try MLX.loadArrays(url: weightsURL)
        
        try module.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: .all
        )
        
        let data = try Data(contentsOf: stateURL)
        let state = try JSONDecoder().decode(State.self, from: data)
        
        dataLoader.seek(to: state.currentLoaderIndex)
        return state.globalStep
    }
}
