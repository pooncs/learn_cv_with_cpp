#include <iostream>
#include "inverse_projection.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);

    double u = 320, v = 240, Z = 5.0;
    cv::Point3d P = reconstruct_point(u, v, Z, K);

    std::cout << "Pixel (" << u << ", " << v << ") at Z=" << Z << "\n";
    std::cout << "3D Point: " << P << "\n";

    return 0;
}
