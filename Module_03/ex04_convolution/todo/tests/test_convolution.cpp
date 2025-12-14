#include <iostream>
#include <cassert>
#include "convolution.hpp"

void test_convolve() {
    cv::Mat img = cv::Mat::zeros(3, 3, CV_8UC1);
    img.at<uchar>(1, 1) = 9;
    
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F);
    
    cv::Mat res = convolve(img, kernel);
    
    // Center pixel: sum of neighbors*1. neighbors are 0, center is 9.
    // 9 * 1 = 9
    assert(res.at<uchar>(1, 1) == 9);
    
    // Top-left neighbor (0,0):
    // Kernel centered at (0,0). Overlap with (1,1) is at kernel(1+1, 1+1) = (2,2).
    // wait, correlation logic:
    // dst(0,0) = sum(src(0+ki, 0+kj) * kernel(ki+1, kj+1))
    // src(1,1) is accessed when ki=1, kj=1. kernel(2,2)=1.
    // So dst(0,0) should be 9.
    assert(res.at<uchar>(0, 0) == 9);

    std::cout << "[PASS] convolve\n";
}

int main() {
    test_convolve();
    return 0;
}
