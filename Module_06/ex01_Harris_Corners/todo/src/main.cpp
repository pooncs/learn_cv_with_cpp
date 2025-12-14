#include <iostream>
#include <opencv2/opencv.hpp>
#include "harris.hpp"

int main(int argc, char** argv) {
    std::cout << "Harris Corner Detection Exercise" << std::endl;
    
    // Load image
    std::string imagePath = (argc > 1) ? argv[1] : "../data/checkerboard.png";
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Image not found, using dummy." << std::endl;
        img = cv::Mat::zeros(200, 200, CV_8UC3);
        cv::rectangle(img, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255, 255, 255), -1);
    }
    
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    cv_curriculum::HarrisConfig config;
    cv::Mat response = cv_curriculum::computeHarrisResponse(gray, config);
    
    // Check if response is empty or all zeros (stub)
    double minVal, maxVal;
    cv::minMaxLoc(response, &minVal, &maxVal);
    if (maxVal == 0) {
        std::cout << "Response is zero. Implement the function!" << std::endl;
        return 0;
    }
    
    float threshold = static_cast<float>(maxVal * 0.01);
    auto corners = cv_curriculum::detectCorners(response, threshold);
    
    std::cout << "Detected " << corners.size() << " corners." << std::endl;
    
    return 0;
}
