#include <iostream>
#include <cassert>
#include "pixel_access.hpp"

void test_process() {
    cv::Mat img = cv::Mat::zeros(10, 10, CV_8UC1);
    
    process_at(img);
    assert(img.at<uchar>(0,0) == 1);
    
    process_ptr(img);
    assert(img.at<uchar>(0,0) == 2);
    
    process_iter(img);
    assert(img.at<uchar>(0,0) == 3);

    std::cout << "[PASS] process_*\n";
}

int main() {
    test_process();
    return 0;
}
