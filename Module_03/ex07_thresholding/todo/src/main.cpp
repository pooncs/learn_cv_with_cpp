#include <iostream>
#include "thresholding.hpp"

int main() {
    // Gradient image with varying brightness
    cv::Mat img(256, 256, CV_8UC1);
    for(int i=0; i<256; ++i) {
        for(int j=0; j<256; ++j) {
            img.at<uchar>(i, j) = (uchar)((i+j)/2);
        }
    }

    cv::Mat binary = my_threshold(img, 127);
    cv::Mat adaptive = my_adaptive_threshold(img, 11, 2);

    std::cout << "Thresholding complete.\n";

    return 0;
}
