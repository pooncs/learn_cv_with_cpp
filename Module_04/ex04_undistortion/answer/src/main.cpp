#include <iostream>
#include "undistort.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);
    
    cv::Mat dist = (cv::Mat_<double>(1, 4) << -0.2, 0.0, 0, 0); // Pincushion/Barrel

    // Create a grid image
    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC1);
    for(int i=0; i<640; i+=40) cv::line(img, cv::Point(i, 0), cv::Point(i, 480), cv::Scalar(255), 1);
    for(int i=0; i<480; i+=40) cv::line(img, cv::Point(0, i), cv::Point(640, i), cv::Scalar(255), 1);

    auto [map_x, map_y] = compute_undistortion_maps(K, dist, img.size());
    
    cv::Mat undistorted;
    cv::remap(img, undistorted, map_x, map_y, cv::INTER_LINEAR);

    std::cout << "Undistortion complete. Use cv::imshow to verify lines are straight/curved.\n";
    
    return 0;
}
