//
//  CoreMLViewController.swift
//  CoreMLTesting
//
//  Created by Michał Ziętera on 06/05/2024.
//

import UIKit
import AVFoundation

final class ONNXRTMPoseViewController: UIViewController {
    
    @IBOutlet weak var captureView: UIView!
    
    private let captureSession = AVCaptureSession()
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    private lazy var detectionLayer = CALayer()
    private var videoSize: CGSize = .zero
    
    private let onnxModelController: ONNXRTMPoseModelController?
    
    required init?(coder: NSCoder) {
        onnxModelController = try? ONNXRTMPoseModelController()
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
            videoDataOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: Int( kCVPixelFormatType_32BGRA)
            ]
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

extension ONNXRTMPoseViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let image = CIImage(cvImageBuffer: pixelBuffer)
        let uiImage = UIImage(ciImage: image)
        do {
            guard let keypointDetections = try onnxModelController?.runModel(onFrame: pixelBuffer) else {
                return
            }
            DispatchQueue.main.async {
                self.detectionLayer.sublayers = nil
                for keypoint in keypointDetections.keypoints {
                    let shapeLayer = CALayer()
                    shapeLayer.borderWidth = 1
                    shapeLayer.borderColor = keypoint.color.cgColor
                    shapeLayer.frame = .init(
                        x: Int(keypoint.x),
                        y: Int(keypoint.y),
                        width: 4,
                        height: 4
                    )
                    self.detectionLayer.addSublayer(shapeLayer)
                }
            }
        } catch {
            print(error)
        }
    }
}
