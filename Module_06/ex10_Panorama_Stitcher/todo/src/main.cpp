#include <iostream>
#include <opencv2/opencv.hpp>
#include "stitcher.hpp"

int main(int argc, char** argv) {
    std::cout << "Panorama Stitcher Exercise" << std::endl;
    
    cv::Mat img1 = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Mat img2 = cv::Mat::zeros(100, 100, CV_8UC3);
    
    auto result = cv_curriculum::stitchImages(img1, img2);
    if (result.empty()) {
        std::cout << "Result empty. Implement the function!" << std::endl;
    }
    
    return 0;
}
