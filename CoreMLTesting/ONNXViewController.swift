//
//  CoreMLViewController.swift
//  CoreMLTesting
//
//  Created by Michał Ziętera on 06/05/2024.
//

import UIKit
import AVFoundation

final class ONNXViewController: UIViewController {
    
    @IBOutlet weak var captureView: UIView!
    
    private let captureSession = AVCaptureSession()
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    private lazy var detectionLayer = CALayer()
    private var videoSize: CGSize = .zero
    
    private let onnxModelController: ONNXModelController?
    
    required init?(coder: NSCoder) {
        onnxModelController = try? ONNXModelController()
        super.init(coder: coder)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
        
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession.stopRunning()
    }
    
    private func setupCamera() {
        setupCameraInput()
        setupCameraOutput()
        setupPreview()
    }
    
    private func setupCameraInput() {
        captureSession.beginConfiguration()
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice), captureSession.canAddInput(videoDeviceInput) else {
            captureSession.commitConfiguration()
            return
        }
        captureSession.sessionPreset = .vga640x480
        
        captureSession.addInput(videoDeviceInput)
        captureSession.commitConfiguration()
        
        do {
            try videoDevice.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions(videoDevice.activeFormat.formatDescription)
            videoSize.width = CGFloat(dimensions.width)
            videoSize.height = CGFloat(dimensions.height)
            videoDevice.unlockForConfiguration()
        } catch {
            print(error)
        }
    }
    
    private func setupCameraOutput() {
        captureSession.beginConfiguration()
        let videoDataOutput = AVCaptureVideoDataOutput()
        let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int( kCVPixelFormatType_32BGRA)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            if let captureConnection = videoDataOutput.connection(with: .video) {
                if captureConnection.isVideoRotationAngleSupported(90) {
                    captureConnection.videoRotationAngle = 90
                }
                captureConnection.isEnabled = true
            }
        }
        captureSession.commitConfiguration()
    }
    
    private func setupPreview() {
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoRotationAngle = 90
        let captureViewLayer = captureView.layer
        previewLayer.frame = captureViewLayer.bounds
        captureViewLayer.addSublayer(previewLayer)
        
        detectionLayer.bounds = .init(origin: .zero, size: .init(width: videoSize.height, height: videoSize.width))
        captureViewLayer.addSublayer(detectionLayer)
        
        let xScale = captureViewLayer.bounds.size.width / videoSize.height
        let yScale = captureViewLayer.bounds.size.height / videoSize.width
        let scale = fmax(xScale, yScale)
        detectionLayer.setAffineTransform(CGAffineTransform(scaleX: scale, y: scale))
        detectionLayer.position = .init(x: captureViewLayer.bounds.midX, y: captureViewLayer.bounds.midY)
    }
}

extension ONNXViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        
//        let testImage = UIImage(named: "wallpapersden.com_tom-hiddleston-man-suit_480x640")!
        let testImage = UIImage(named: "size0-full")!
//        let testImage = UIImage(named: "test")!
        
        guard let pixelBuffer = testImage.pixelBuffer(width: 640, height: 640) else {
            return
        }
        
//        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
//            return
//        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let image = CIImage(cvImageBuffer: pixelBuffer)
        let uiImage = UIImage(ciImage: image)
        do {
            let detections = try onnxModelController?.runModel(onFrame: pixelBuffer) ?? []
            print(detections)
            DispatchQueue.main.async {
                self.detectionLayer.sublayers = nil
                for detection in detections {
                    let shapeLayer = CALayer()
                    shapeLayer.borderWidth = 2
                    shapeLayer.borderColor = UIColor.red.cgColor
                    shapeLayer.frame = detection.boundingBox
                    
                    let textLayer = CATextLayer()
                    textLayer.foregroundColor = UIColor.red.cgColor
                    textLayer.font = UIFont.systemFont(ofSize: 12)
                    textLayer.fontSize = 12
                    textLayer.contentsScale = self.view.window?.windowScene?.screen.scale ?? 1.0
                    textLayer.string = String(format: "%.1f", detection.confidence * 100) + "%"
                    textLayer.frame = shapeLayer.frame
                    self.detectionLayer.addSublayer(shapeLayer)
                    self.detectionLayer.addSublayer(textLayer)
                }
            }
        } catch {
            print(error)
        }
    }
}
