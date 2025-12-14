#include <iostream>
#include <cassert>
#include "color_spaces.hpp"

void test_gray() {
    cv::Mat img(1, 1, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue
    cv::Mat gray = to_gray(img);
    // 0.114 * 255 = 29.07 -> 29
    assert(std::abs(gray.at<uchar>(0,0) - 29) <= 1);
    std::cout << "[PASS] to_gray\n";
}

int main() {
    test_gray();
    return 0;
}
