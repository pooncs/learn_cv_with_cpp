#include <iostream>
#include "pinhole.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);

    cv::Mat points_3d = (cv::Mat_<double>(2, 3) << 
        0, 0, 10,   // Center
        10, 10, 20  // Top-Right far
    );

    cv::Mat points_2d = project_points(points_3d, K);

    std::cout << "3D Points:\n" << points_3d << "\n";
    std::cout << "Projected 2D Points:\n" << points_2d << "\n";

    return 0;
}
