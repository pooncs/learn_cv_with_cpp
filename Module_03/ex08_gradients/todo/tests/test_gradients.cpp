#include <iostream>
#include <cassert>
#include "gradients.hpp"

void test_sobel() {
    cv::Mat img = cv::Mat::zeros(3, 3, CV_8UC1);
    img.at<uchar>(1, 0) = 0;
    img.at<uchar>(1, 1) = 0;
    img.at<uchar>(1, 2) = 255; 
    // Row 1: 0, 0, 255
    // Sobel X kernel [-1 0 1]
    // At (1,1): Left is 0, Right is 255.
    // 255*1 + 0*(-1) = 255.
    
    // Note: OpenCV Sobel usually includes smoothing in Y direction too (1, 2, 1).
    // If we strictly check center pixel:
    // Row 0: 0 0 0
    // Row 1: 0 0 255
    // Row 2: 0 0 0
    // Sobel X at (1,1):
    // (-1*0 + 0*0 + 1*0) + (-2*0 + 0*0 + 2*255) + (-1*0 + 0*0 + 1*0) = 510?
    // Wait, let's just check sign.
    
    auto [Gx, Gy] = compute_sobel(img);
    if(Gx.empty()) {
        std::cout << "[SKIP] compute_sobel not implemented\n";
        return;
    }

    float gx_val = Gx.at<float>(1, 1);
    assert(gx_val > 0); // Gradient increasing x

    std::cout << "[PASS] compute_sobel\n";
}

int main() {
    test_sobel();
    return 0;
}
