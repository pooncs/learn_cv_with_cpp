#include <iostream>
#include "inverse_projection.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);

    double u = 320, v = 240, Z = 5.0;
    // TODO: Call reconstruct_point
    
    return 0;
}
