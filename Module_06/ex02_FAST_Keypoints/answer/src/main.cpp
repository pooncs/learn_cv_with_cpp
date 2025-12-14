#include <iostream>
#include <opencv2/opencv.hpp>
#include "fast.hpp"

int main(int argc, char** argv) {
    const cv::String keys =
        "{help h usage ? |      | print this message   }"
        "{@image         |      | image for processing }";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    std::string imagePath = parser.get<std::string>("@image");
    if (imagePath.empty()) {
        imagePath = "../data/checkerboard.png"; 
    }

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
         std::cout << "Image not found at " << imagePath << ", creating synthetic image." << std::endl;
        img = cv::Mat::zeros(400, 400, CV_8UC3);
        cv::rectangle(img, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 255, 255), -1);
    }
    
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();
    
    cv_curriculum::FastConfig config;
    config.threshold = 40;
    
    // My Implementation
    auto kps = cv_curriculum::detectFAST(gray, config);
    std::cout << "Detected " << kps.size() << " keypoints (Manual)." << std::endl;
    
    cv::Mat outImg;
    cv::drawKeypoints(img, kps, outImg, cv::Scalar(0, 0, 255));
    
    // OpenCV Implementation for comparison
    std::vector<cv::KeyPoint> cvKps;
    cv::FAST(gray, cvKps, config.threshold, true);
    std::cout << "Detected " << cvKps.size() << " keypoints (OpenCV)." << std::endl;

    cv::Mat cvOutImg;
    cv::drawKeypoints(img, cvKps, cvOutImg, cv::Scalar(0, 255, 0));
    
    cv::imwrite("fast_result_manual.png", outImg);
    cv::imwrite("fast_result_opencv.png", cvOutImg);
    
    return 0;
}
