#include <iostream>
#include "distortion.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);
    
    cv::Mat dist = (cv::Mat_<double>(1, 4) << -0.1, 0.01, 0, 0); // Barrel distortion

    cv::Point3d P(10, 10, 20);
    cv::Point2d uv = project_distorted(P, K, dist);

    std::cout << "Projected with distortion: " << uv << "\n";

    return 0;
}
