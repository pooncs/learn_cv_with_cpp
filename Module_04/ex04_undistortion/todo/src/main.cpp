#include <iostream>
#include "undistort.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);
    
    cv::Mat dist = (cv::Mat_<double>(1, 4) << -0.2, 0.0, 0, 0); // Pincushion/Barrel

    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC1);
    // TODO: Call compute_undistortion_maps
    
    return 0;
}
