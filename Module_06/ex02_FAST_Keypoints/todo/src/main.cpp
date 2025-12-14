#include <iostream>
#include <opencv2/opencv.hpp>
#include "fast.hpp"

int main(int argc, char** argv) {
    std::cout << "FAST Keypoint Detection Exercise" << std::endl;
    
    std::string imagePath = (argc > 1) ? argv[1] : "../data/checkerboard.png";
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        img = cv::Mat::zeros(400, 400, CV_8UC3);
        cv::rectangle(img, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 255, 255), -1);
    }
    
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    cv_curriculum::FastConfig config;
    auto kps = cv_curriculum::detectFAST(gray, config);
    
    if (kps.empty()) {
        std::cout << "No keypoints detected. Implement the function!" << std::endl;
    } else {
        std::cout << "Detected " << kps.size() << " keypoints." << std::endl;
        cv::drawKeypoints(img, kps, img, cv::Scalar(0, 0, 255));
        cv::imwrite("fast_todo_result.png", img);
    }
    
    return 0;
}
