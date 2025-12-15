#include <iostream>
#include "colorize.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 500, 0, 320, 0, 500, 240, 0, 0, 1);
    
    // Create a Red image
    cv::Mat img = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 255)); // Red in BGR
    
    std::vector<cv::Point3f> cloud;
    // TODO: Create points
    // TODO: Call colorize_cloud

    return 0;
}
