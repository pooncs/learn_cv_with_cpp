#include <iostream>
#include "homography.hpp"

int main() {
    // Define a square in src
    std::vector<cv::Point2f> src = {
        {0, 0}, {100, 0}, {100, 100}, {0, 100}
    };
    
    // Define a distorted shape in dst
    std::vector<cv::Point2f> dst = {
        {10, 10}, {90, 20}, {110, 80}, {-10, 90}
    };

    // TODO: Call compute_homography and warp_image
    
    return 0;
}
