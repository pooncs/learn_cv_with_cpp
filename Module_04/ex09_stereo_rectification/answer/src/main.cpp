#include <iostream>
#include "stereo.hpp"

int main() {
    // Synthetic calibration data
    cv::Mat K1 = (cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
    cv::Mat K2 = K1.clone();
    cv::Mat D1 = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat D2 = cv::Mat::zeros(1, 5, CV_64F);
    
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat T = (cv::Mat_<double>(3, 1) << -100, 0, 0); // 100mm baseline

    cv::Size img_size(640, 480);
    
    StereoMaps maps = compute_stereo_rectification(K1, D1, K2, D2, R, T, img_size);

    std::cout << "Q Matrix:\n" << maps.Q << "\n";
    std::cout << "Rectification maps computed.\n";

    return 0;
}
