//
//  ONNXModelController.swift
//  CoreMLTesting
//
//  Created by Michał Ziętera on 10/05/2024.
//

import Accelerate
import onnxruntime_objc

final class ONNXRTMPoseModelController {
    
    enum OrtModelError: Error {
        case error(_ message: String)
    }
    
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 192
    let inputHeight = 256
    let threshold: Float = 0.9
    let keypointCount = 26
    let horizontalHeatmapResolution = 384
    let verticalHeatmapResolution = 512
    
    private let colors: [UIColor] = [
        .initWith(red: 0, green: 0, blue: 0),
        .initWith(red: 255, green: 255, blue: 255),
        .initWith(red: 255, green: 0, blue: 0),
        .initWith(red: 0, green: 255, blue: 0),
        .initWith(red: 0, green: 0, blue: 255),
        .initWith(red: 255, green: 255, blue: 0),
        .initWith(red: 255, green: 0, blue: 255),
        .initWith(red: 128, green: 0, blue: 0),
        .initWith(red: 0, green: 128, blue: 0),
        .initWith(red: 0, green: 0, blue: 128),
        .initWith(red: 128, green: 128, blue: 0), // 10
        .initWith(red: 128, green: 0, blue: 128),
        .initWith(red: 0, green: 128, blue: 128),
        .initWith(red: 153, green: 153, blue: 255),
        .initWith(red: 153, green: 51, blue: 102),
        .initWith(red: 255, green: 255, blue: 204),
        .initWith(red: 0, green: 102, blue: 204),
        .initWith(red: 255, green: 102, blue: 0),
        .initWith(red: 51, green: 153, blue: 102),
        .initWith(red: 128, green: 0, blue: 128),
        .initWith(red: 0, green: 204, blue: 255),
        .initWith(red: 255, green: 153, blue: 255),
        .initWith(red: 51, green: 51, blue: 51),
        .initWith(red: 153, green: 51, blue: 0),
        .initWith(red: 150, green: 150, blue: 150),
        .initWith(red: 0, green: 51, blue: 0),
        .initWith(red: 204, green: 255, blue: 204) // 26
        
    ]
    
    private var session: ORTSession?
    
    private var labels: [String] = []
    
    init() throws {
        Task {
            guard let modelPath = Bundle.main.path(forResource: "rtmpose-m", ofType: "onnx") else {
                throw OrtModelError.error("Could not load model")
            }
            
            
            let ortEnv = try ORTEnv(loggingLevel: .info)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(.verbose)
            try options.setGraphOptimizationLevel(.none)
            try options.setIntraOpNumThreads(6)
//            try options.addConfigEntry(withKey: "session.dynamic_block_base", value: "4")
            let coreMLOptions = ORTCoreMLExecutionProviderOptions()
            try options.appendCoreMLExecutionProvider(with: coreMLOptions)
            session = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: options)
        }
    }
    
    func runModel(onFrame pixelBuffer: CVPixelBuffer) throws -> KeypointsDetection? {
        guard let session else { return nil }
        
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        let imageChannels = 4
        assert(imageChannels >= inputChannels)
        
        let imageWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let imageHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = preprocess(ofSize: scaledSize, pixelBuffer) else {
            return nil
        }
        
        let image = CIImage(cvImageBuffer: pixelBuffer)
        let uiImage = UIImage(ciImage: image)
        
        let scaled = CIImage(cvImageBuffer: scaledPixelBuffer)
        let scaleduiImage = UIImage(ciImage: scaled)
        
        let inputName = "input"
        
        guard let rgbData = rgbDataFromBuffer(
            scaledPixelBuffer,
            isModelQuantized: false
        ) else {
            print("Failed to convert the image buffer to RGB data.")
            return nil
        }
        
        
//        let imageBlob = OpenCVBridge.convertToBlob(from: scaleduiImage)!
//        let opencvImageBlob = OpenCVBridge.convertToBlob(from: pixelBuffer)!
//        let imageBlob = Data(copyingBufferOf: [UInt8](opencvImageBlob).map { Float($0) / 255.0 })
        
//        let first = Array(rgbData)
//        let second = Array(imageBlob)
        
        let inputShape: [NSNumber] = [batchSize as NSNumber,
                                      inputChannels as NSNumber,
                                      inputHeight as NSNumber,
                                      inputWidth as NSNumber]
        
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: rgbData),
                                       elementType: ORTTensorElementDataType.float,
                                       shape: inputShape)
        
        let outputs = try session.run(withInputs: [inputName: inputTensor],
                                      outputNames: ["simcc_x", "simcc_y"],
                                      runOptions: nil)
        
        guard let rawSimccxOutputValue = outputs["simcc_x"] else {
            throw OrtModelError.error("failed to get model output")
        }
        
        guard let rawSimccyOutputValue = outputs["simcc_y"] else {
            throw OrtModelError.error("failed to get model output")
        }
        
        let rawSimccxOutputData = try rawSimccxOutputValue.tensorData() as Data
        let rawSimccyOutputData = try rawSimccyOutputValue.tensorData() as Data
        guard
            let outputSimccxArr: [Float] = Array(unsafeData: rawSimccxOutputData),
            let outputSimccyArr: [Float] = Array(unsafeData: rawSimccyOutputData)
        else {
            return nil
        }
        
        let horizontalScale = 0.5 / Float(inputWidth)
        let verticalScale = 0.5 / Float(inputHeight)
        var confidenceArray = [Float]()
        var positionArray = [(x: Float, y: Float, color: UIColor)]()
        
        for keypoint in 0..<keypointCount {
            let xStartIndex = keypoint * horizontalHeatmapResolution
            let yStartIndex = keypoint * verticalHeatmapResolution
            let currentXrow = outputSimccxArr[xStartIndex..<(xStartIndex + horizontalHeatmapResolution)]
            let currentYrow = outputSimccyArr[yStartIndex..<(yStartIndex + verticalHeatmapResolution)]
            
            guard let currentXMaxValue = findMaxAndIndex(in: Array(currentXrow)),
                  let currentYMaxValue = findMaxAndIndex(in: Array(currentYrow)) else {
                return nil
            }
            positionArray.append(
                (
                    x: Float(currentXMaxValue.index) * horizontalScale * Float(imageWidth),
                    y: Float(currentYMaxValue.index) * verticalScale * Float(imageHeight),
                    color: colors[keypoint]
                )
            )
            let confidence = min(min(currentXMaxValue.confidence, 1.0), currentYMaxValue.confidence)
            confidenceArray.append(confidence)
        }
        return .init(keypoints: positionArray, confidenceScores: confidenceArray)
    }
    
    private func findMaxAndIndex(in row: [Float]) -> (confidence: Float, index: Int)? {
       row.enumerated()
            .max { first, second in
                first.element < second.element
            }.map { ($0.element, $0.offset) }
    }
    
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        isModelQuantized: Bool = true,
        groupColorComponents: Bool = true
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)
        
        switch pixelBufferFormat {
        case kCVPixelFormatType_32BGRA:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32ARGB:
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32RGBA:
            vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        default:
            // Unknown pixel format.
            return nil
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        
        if isModelQuantized { return byteData }
        
        let imageBytes = [UInt8](byteData)
        let maxRGBValue: Float32 = 255.0
        
        if groupColorComponents {
            var index = 0
            var red = [UInt8]()
            var green = [UInt8]()
            var blue = [UInt8]()
            while index < imageBytes.count {
                red.append(imageBytes[index])
                green.append(imageBytes[index+1])
                blue.append(imageBytes[index+2])
                index += 3
            }
            let combined = blue + green + red
            
            let blueModified = blue.map { (Float($0) - 123.675) / 58.395 }
            let greenModified = green.map { (Float($0) - 116.28) / 57.12 }
            let redModified = red.map { (Float($0) - 103.53) / 57.375 }
            let combinedModified = blueModified + greenModified + redModified
//            return Data(copyingBufferOf: combined.map { Float($0) / maxRGBValue })
            return Data(copyingBufferOf: combinedModified)
        } else {
            let converted = Data(copyingBufferOf: imageBytes.map { Float($0) / maxRGBValue }) // 0-255 to 0-1
            return converted
        }
    }
    
    // This method preprocesses the image by cropping pixel buffer to biggest square
    // and scaling the cropped image to model dimensions.
    private func preprocess(
        ofSize size: CGSize,
        _ buffer: CVPixelBuffer
    ) -> CVPixelBuffer? {
        let imageWidth = CVPixelBufferGetWidth(buffer)
        let imageHeight = CVPixelBufferGetHeight(buffer)
        let pixelBufferType = CVPixelBufferGetPixelFormatType(buffer)
        
        assert(pixelBufferType == kCVPixelFormatType_32BGRA ||
               pixelBufferType == kCVPixelFormatType_32ARGB)
        
        let inputImageRowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let imageChannels = 4
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        // Find the biggest square in the pixel buffer and advance rows based on it.
        guard let inputBaseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        // Get vImage_buffer
        var inputVImageBuffer = vImage_Buffer(data: inputBaseAddress,
                                              height: UInt(imageHeight),
                                              width: UInt(imageWidth),
                                              rowBytes: inputImageRowBytes)
        
        let scaledRowBytes = Int(size.width) * imageChannels
        guard let scaledImageBytes = malloc(Int(size.height) * scaledRowBytes) else {
            return nil
        }
        
        var scaledVImageBuffer = vImage_Buffer(data: scaledImageBytes,
                                               height: UInt(size.height),
                                               width: UInt(size.width),
                                               rowBytes: scaledRowBytes)
        
        // Perform the scale operation on input image buffer and store it in scaled vImage buffer.
        let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &scaledVImageBuffer, nil, vImage_Flags(0))
        
        guard scaleError == kvImageNoError else {
            free(scaledImageBytes)
            return nil
        }
        
        let releaseCallBack: CVPixelBufferReleaseBytesCallback = { _, pointer in
            
            if let pointer = pointer {
                free(UnsafeMutableRawPointer(mutating: pointer))
            }
        }
        
        var scaledPixelBuffer: CVPixelBuffer?
        
        // Convert the scaled vImage buffer to CVPixelBuffer
        let conversionStatus = CVPixelBufferCreateWithBytes(
            nil, Int(size.width), Int(size.height), pixelBufferType, scaledImageBytes,
            scaledRowBytes, releaseCallBack, nil, nil, &scaledPixelBuffer
        )
        
        guard conversionStatus == kCVReturnSuccess else {
            free(scaledImageBytes)
            return nil
        }
        
        return scaledPixelBuffer
    }
}

private extension Array {
    // Create a new array from the bytes of the given unsafe data.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    }
}


private extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    ///
    /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
    ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
    ///     data from the resulting buffer has undefined behavior.
    /// - Parameter array: An array with elements of type `T`.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

