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
    
    // Convert to meters (0.001)
    auto cloud = depth_to_cloud(depth, K, 0.001f);
    
    std::cout << "Generated cloud with " << cloud.size() << " points.\n";
    if(!cloud.empty()) {
        std::cout << "Point 0: " << cloud[0].x << " " << cloud[0].y << " " << cloud[0].z << "\n";
    }

    return 0;
}
