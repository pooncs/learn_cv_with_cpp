#include <iostream>
#include <opencv2/opencv.hpp>
#include "orb_custom.hpp"

int main(int argc, char** argv) {
    std::cout << "ORB Descriptor Extraction Exercise" << std::endl;
    
    std::string imagePath = (argc > 1) ? argv[1] : "../data/checkerboard.png";
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        img = cv::Mat::zeros(400, 400, CV_8UC3);
        cv::line(img, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 255, 255), 2);
    }
    
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    // Detect
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(gray, keypoints, 20, true);
    
    if (keypoints.empty()) {
        std::cout << "No keypoints detected." << std::endl;
        return 0;
    }
    
    // Compute Orientation
    cv_curriculum::computeOrientation(gray, keypoints);
    
    // Check if orientation was computed (check if all angles are 0, unlikely if implemented)
    float sumAngles = 0;
    for (const auto& kp : keypoints) sumAngles += kp.angle;
    if (sumAngles == 0 && keypoints.size() > 10) {
        std::cout << "Warning: All angles are 0. Orientation computation might be missing." << std::endl;
    }
    
    // Extract
    cv_curriculum::OrbConfig config;
    cv::Mat descriptors = cv_curriculum::extractOrbDescriptors(gray, keypoints, config);
    
    if (descriptors.empty()) {
        std::cout << "Descriptors are empty. Implement the function!" << std::endl;
    } else {
        std::cout << "Extracted descriptors: " << descriptors.size() << std::endl;
    }
    
    return 0;
}
