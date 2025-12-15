#include <iostream>
#include "depth_util.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 500, 0, 320, 0, 500, 240, 0, 0, 1);
    
    // Create a synthetic depth ramp
    cv::Mat depth = cv::Mat::zeros(480, 640, CV_16U);
    for(int v=0; v<480; ++v) {
        for(int u=0; u<640; ++u) {
            depth.at<uint16_t>(v, u) = (uint16_t)(1000 + u); // 1m to 1.6m
        }
    }
    
    // TODO: Call depth_to_cloud

    return 0;
}
