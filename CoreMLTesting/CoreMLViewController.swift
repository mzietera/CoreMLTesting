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
        
    }
    
    private func setupVision() {
//        yolov8n.mlpackage
//        guard let modelUrl = Bundle.main.url(forResource: "yolov8n", withExtension: "mlpackage") else {
//            present(UIAlertController(title: "Could not load model", message: nil, preferredStyle: .alert), animated: true)
//            return
//        }
        do {
            let yolov8nModel = try yolov8n()
            let visionModel = try VNCoreMLModel(for: yolov8nModel.model)
            let objectRecognition = VNCoreMLRequest(model: visionModel) { request, error in
                
            }
        } catch {
            present(UIAlertController(title: error.localizedDescription, message: nil, preferredStyle: .alert), animated: true)
        }
    }
}

extension CoreMLViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
}
