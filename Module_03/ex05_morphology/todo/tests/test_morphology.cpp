#include <iostream>
#include <cassert>
#include "morphology.hpp"

void test_dilate() {
    // 3x3 image with center dot
    cv::Mat img = cv::Mat::zeros(3, 3, CV_8UC1);
    img.at<uchar>(1, 1) = 255;
    
    cv::Mat res = my_dilate(img);
    // All pixels should become 255 because center is 255 and it's 3x3
    // (Each pixel has (1,1) as neighbor)
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            assert(res.at<uchar>(i, j) == 255);
            
    std::cout << "[PASS] my_dilate\n";
}

void test_erode() {
    // 3x3 all white
    cv::Mat img = cv::Mat::ones(3, 3, CV_8UC1) * 255;
    // Set corner to 0
    img.at<uchar>(0, 0) = 0;
    
    cv::Mat res = my_erode(img);
    // Center pixel (1,1) has (0,0) as neighbor, which is 0. So center should be 0.
    assert(res.at<uchar>(1, 1) == 0);
    
    std::cout << "[PASS] my_erode\n";
}

int main() {
    test_dilate();
    test_erode();
    return 0;
}
