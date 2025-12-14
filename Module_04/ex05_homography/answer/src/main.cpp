#include <iostream>
#include "homography.hpp"

int main() {
    // Define a square in src
    std::vector<cv::Point2f> src = {
        {0, 0}, {100, 0}, {100, 100}, {0, 100}
    };
    
    // Define a distorted shape in dst (trapezoid)
    std::vector<cv::Point2f> dst = {
        {10, 10}, {90, 20}, {110, 80}, {-10, 90}
    };

    cv::Mat H = compute_homography(src, dst);
    std::cout << "Homography Matrix:\n" << H << "\n";

    // Warp a test image
    cv::Mat img = cv::Mat::zeros(200, 200, CV_8UC3);
    cv::rectangle(img, cv::Rect(0, 0, 100, 100), cv::Scalar(0, 255, 0), cv::FILLED);
    
    cv::Mat warped = warp_image(img, H, cv::Size(200, 200));
    
    std::cout << "Image warped.\n";

    return 0;
}
