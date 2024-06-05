//
//  UIColor+init.swift
//  CoreMLTesting
//
//  Created by Michał Ziętera on 23/05/2024.
//

import UIKit

extension UIColor {
    class func initWith(red: Int, green: Int, blue: Int) -> UIColor {
        .init(red: CGFloat(red) / 255.0, green: CGFloat(green) / 255.0, blue: CGFloat(blue) / 255.0, alpha: 1)
    }
}
