#include <iostream>
#include "colorize.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 500, 0, 320, 0, 500, 240, 0, 0, 1);
    
    // Create a Red image
    cv::Mat img = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 255)); // Red in BGR
    
    // Create points along Z axis (0,0,Z) -> projects to center (320, 240)
    std::vector<cv::Point3f> cloud;
    for(int i=1; i<=10; ++i) cloud.push_back({0, 0, (float)i});
    
    auto colored = colorize_cloud(cloud, img, K);
    
    std::cout << "Colored " << colored.size() << " points.\n";
    if(!colored.empty()) {
        std::cout << "Point 0 color: " << (int)colored[0].r << " " << (int)colored[0].g << " " << (int)colored[0].b << "\n";
    }

    return 0;
}
