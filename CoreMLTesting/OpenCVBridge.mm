////
////  OpenCVBridge.m
////  CoreMLTesting
////
////  Created by Michał Ziętera on 22/05/2024.
////
//
//#import <opencv2/opencv.hpp>
//#import <opencv2/imgcodecs/ios.h>
//#import "OpenCVBridge.h"
//#import <UIKit/UIKit.h>
//
//@implementation OpenCVBridge
//
//+ (NSData *) convertToBlobFrom: (UIImage *) image {
//    cv::Mat imageToMat;
////    imageToMat = cv::Mat(256, 192, CV_8UC3);
//    CGImageAlphaInfo aInfo = CGImageGetAlphaInfo(image.CGImage);
//    UIImageToMat(image, imageToMat, aInfo != kCGImageAlphaNone);
//    UIImage * returnedImage = MatToUIImage(imageToMat);
//    cv::Mat imageAsBlob = cv::dnn::blobFromImage(imageToMat, 1.0 / 127.5, cv::Size(192, 256), cv::Scalar(123.675, 116.28, 103.53), true);
//    return [NSData dataWithBytes:imageAsBlob.data length:imageAsBlob.elemSize()*imageAsBlob.total()];
//}
//
//+ (NSData *) convertToBlobFromPixelBuffer:(CVPixelBufferRef) pixelBuffer {
//    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
//    int bufferWidth = (int)CVPixelBufferGetWidth(pixelBuffer);
//    int bufferHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
////    unsigned char *pixel = (unsigned char *)CVPixelBufferGetBaseAddress(pixelBuffer);
//    
//    cv::Mat mat = cv::Mat(bufferHeight, bufferWidth, CV_8UC4, CVPixelBufferGetBytesPerRow(pixelBuffer));
//    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
//    cv::Mat imageAsBlob = cv::dnn::blobFromImage(mat, 1.0 / 127.5, cv::Size(192, 256), cv::Scalar(123.675, 116.28, 103.53), true);
//    return [NSData dataWithBytes:imageAsBlob.data length:imageAsBlob.elemSize()*imageAsBlob.total()];
//}
//
//@end
