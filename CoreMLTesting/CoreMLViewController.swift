//
//  CoreMLViewController.swift
//  CoreMLTesting
//
//  Created by Michał Ziętera on 06/05/2024.
//

import UIKit
import Vision
import CoreML
import AVFoundation

final class CoreMLViewController: UIViewController {
    
    @IBOutlet weak var captureView: UIView!
    
    private let captureSession = AVCaptureSession()
    
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    private var vnRequest: VNRequest!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupVision()
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
    }
    
    private func setupCameraOutput() {
        captureSession.beginConfiguration()
        let videoDataOutput = AVCaptureVideoDataOutput()
        let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
//            if let videoConnection = videoDataOutput.connection(with: .video) {
//                if videoConnection.isVideoRotationAngleSupported(0) {
//                    videoConnection.videoRotationAngle = 0
//                }
//            }
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        }
        captureSession.commitConfiguration()
    }
    
    private func setupPreview() {
//        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        let captureViewLayer = captureView.layer
        previewLayer.frame = captureViewLayer.bounds
        captureViewLayer.addSublayer(previewLayer)
        
//        let detectionLayer =
    }
    
    private func setupVision() {
        do {
            let yolov8nModel = try yolov8n()
            let visionModel = try VNCoreMLModel(for: yolov8nModel.model)
            vnRequest = VNCoreMLRequest(model: visionModel)
        } catch {
            present(UIAlertController(title: error.localizedDescription, message: nil, preferredStyle: .alert), animated: true)
        }
    }
    
    private func process(vnObservations: [VNRecognizedObjectObservation]) {
        for object in vnObservations {
            print("\(object.labels[0].identifier) \(object.confidence)")
        }
    }
}

extension CoreMLViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        do {
            try imageRequestHandler.perform([vnRequest])
            if let recognizedObjects = vnRequest.results as? [VNRecognizedObjectObservation] {
                process(vnObservations: recognizedObjects)
            }
        } catch {
            print(error)
        }
    }
}
