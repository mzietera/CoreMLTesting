//
//  CGRect+flippedVertically.swift
//  CoreMLTesting
//
//  Created by Michał Ziętera on 07/05/2024.
//

import Foundation

extension CGRect {
    var flippedVertically: CGRect {
        return CGRect(x: minX, y: 1-maxY, width: width, height: height)
    }
}
