//
//  Detection.swift
//  CoreMLTesting
//
//  Created by Michał Ziętera on 06/05/2024.
//

import Foundation

struct Detection {
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}
