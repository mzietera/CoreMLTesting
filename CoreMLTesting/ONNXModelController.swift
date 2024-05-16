//
//  ONNXModelController.swift
//  CoreMLTesting
//
//  Created by Michał Ziętera on 10/05/2024.
//

import Accelerate
import onnxruntime_objc

final class ONNXModelController {
    
    enum OrtModelError: Error {
        case error(_ message: String)
    }
    
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 640
    let inputHeight = 640
    let threshold: Float = 0.9
    
    private var session: ORTSession?
    
    private var labels: [String] = []
    
    init() throws {
        Task {
            guard let modelPath = Bundle.main.path(forResource: "yolov5n", ofType: "onnx") else {
                throw OrtModelError.error("Could not load model")
            }
            
            
            let ortEnv = try ORTEnv(loggingLevel: .info)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(.verbose)
            let coreMLOptions = ORTCoreMLExecutionProviderOptions()
            try options.appendCoreMLExecutionProvider(with: coreMLOptions)
            session = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: options)
            //        loadLabels()
        }
    }
    
    func runModel(onFrame pixelBuffer: CVPixelBuffer) throws -> [Detection] {
        guard let session else { return [] }
        
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        let imageChannels = 4
        assert(imageChannels >= inputChannels)
        
//        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
//        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
//        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
//        guard let scaledPixelBuffer = preprocess(ofSize: scaledSize, pixelBuffer) else {
//            return []
//        }
        
        let inputName = "images"
        
        guard let rgbData = rgbDataFromBuffer(
            pixelBuffer,
            isModelQuantized: false
        ) else {
            print("Failed to convert the image buffer to RGB data.")
            return []
        }
        
        let inputShape: [NSNumber] = [batchSize as NSNumber,
                                      inputChannels as NSNumber,
                                      inputHeight as NSNumber,
                                      inputWidth as NSNumber]
        
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: rgbData),
                                       elementType: ORTTensorElementDataType.float,
                                       shape: inputShape)
        
        let outputs = try session.run(withInputs: [inputName: inputTensor],
                                      outputNames: ["output0"],
                                      runOptions: nil)
        
        guard let rawOutputValue = outputs["output0"] else {
            throw OrtModelError.error("failed to get model output")
        }
        
        let rawOutputData = try rawOutputValue.tensorData() as Data
        guard let outputArr: [Float32] = Array(unsafeData: rawOutputData) else {
            return []
        }
        return postprocess(output: outputArr)
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
//            let byteData = combined.map { Float($0) / maxRGBValue }.prefix(1280)
//            print(byteData)
//            print(combined.prefix(1280))
            
            
            var blueTransposed = [UInt8]()
            var greenTransposed = [UInt8]()
            var redTransposed = [UInt8]()
            
            for i in 0...639 {
                var rowIndex = i
                while rowIndex < 640*640 {
                    blueTransposed.append(blue[rowIndex])
                    greenTransposed.append(green[rowIndex])
                    redTransposed.append(red[rowIndex])
                    rowIndex += 640
                }
            }
//            let combined = blue + green + red
            let combined = redTransposed + greenTransposed + blueTransposed
            
            return Data(copyingBufferOf: combined.map { Float($0) / maxRGBValue })
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
    
    func postprocess(output: [Float32]) -> [Detection] {
        // filter out bounding boxes for person and with confidence greater than threshold
        var boxes = [BoundingBox]()
        for i in 0...(25200 - 1) {
//            let objectness = output[i*85 + 4]
//            let personClassConfidence = output[i*85 + 5]
//            let confidence = objectness * personClassConfidence
            let confidence = output[i*85 + 5]
//            print(output[i*85...i*85+5])
//            print("index \(i): \(output[i*85]) \(output[i*85 + 1]) \(output[i*85 + 2]) \(output[i*85 + 3]) \(output[i*85 + 5])")
            
            
            if confidence >= threshold {
                let midX = CGFloat(output[i*85])
                let midY = CGFloat(output[i*85 + 1])
                let width = CGFloat(output[i*85 + 2])
                let height = CGFloat(output[i*85 + 3])
                boxes.append(.init(
                    classIndex: 0,
                    score: confidence,
                    rect: .init(
                        x: midX - width / 2,
                        y: midY - height / 2,
                        width: width,
                        height: height
                    ))
                )
                
//                let minX = CGFloat(output[i*85])
//                let minY = CGFloat(output[i*85 + 1])
//                let maxX = CGFloat(output[i*85 + 2])
//                let maxY = CGFloat(output[i*85 + 3])
//                boxes.append(.init(
//                    classIndex: 0,
//                    score: confidence,
//                    rect: .init(
//                        x: minX,
//                        y: minY,
//                        width: maxX - minX,
//                        height: maxY - minY
//                    ))
//                )
            }
        }
        
        let indices = nonMaxSuppression(boundingBoxes: boxes, iouThreshold: 0.45, maxBoxes: 100)
        var detections = [Detection]()
        for index in indices {
            detections.append(.init(
                label: "person",
                confidence: boxes[index].score,
                boundingBox: boxes[index].rect
            ))
        }

        return detections
    }

//    private func loadLabels() {
//        guard let fileUrl = Bundle.main.url(forResource: "yolov8nLabels", withExtension: "txt") else {
//            print("Label files not found")
//            return
//        }
//        do {
//            let contents = try String(contentsOf: fileUrl, encoding: .utf8)
//            labels = contents.components(separatedBy: .newlines)
//        } catch {
//            print("Labels file cannot be read")
//        }
//    }
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

