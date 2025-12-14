#include <iostream>
#include "convolution.hpp"

int main() {
    // 5x5 image with a dot in center
    cv::Mat img = cv::Mat::zeros(5, 5, CV_8UC1);
    img.at<uchar>(2, 2) = 100;

    // Box blur kernel (all 1/9)
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0f;

    cv::Mat res = convolve(img, kernel);
    
    if(!res.empty()) {
        std::cout << "Input:\n" << img << "\n\n";
        std::cout << "Kernel:\n" << kernel << "\n\n";
        std::cout << "Output:\n" << res << "\n\n";
    } else {
        std::cout << "Not implemented yet.\n";
    }

    return 0;
}
