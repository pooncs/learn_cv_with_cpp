#include <iostream>
#include <cassert>
#include "thresholding.hpp"

void test_thresh() {
    cv::Mat img(1, 2, CV_8UC1);
    img.at<uchar>(0,0) = 100;
    img.at<uchar>(0,1) = 200;
    
    cv::Mat res = my_threshold(img, 150);
    assert(res.at<uchar>(0,0) == 0);
    assert(res.at<uchar>(0,1) == 255);
    std::cout << "[PASS] my_threshold\n";
}

int main() {
    test_thresh();
    return 0;
}
