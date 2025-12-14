#include <gtest/gtest.h>
#include "undistort.hpp"

TEST(UndistortTest, NoDistortion) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        100, 0, 50,
        0, 100, 50,
        0, 0, 1);
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
    cv::Size size(100, 100);
    
    auto [mx, my] = compute_undistortion_maps(K, dist, size);
    
    // With no distortion, map(u,v) should be (u,v)
    EXPECT_NEAR(mx.at<float>(50, 50), 50.0, 1e-4);
    EXPECT_NEAR(my.at<float>(50, 50), 50.0, 1e-4);
}
