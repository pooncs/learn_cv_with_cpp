#include <gtest/gtest.h>
#include "depth_util.hpp"

TEST(DepthTest, BackProjection) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        100, 0, 50,
        0, 100, 50,
        0, 0, 1);
    
    cv::Mat depth = cv::Mat::zeros(1, 1, CV_32F);
    depth.at<float>(0, 0) = 10.0f; // Center pixel at 10m
    // Pixel (0,0) corresponds to u=0, v=0.
    // cx=50, cy=50. fx=100, fy=100.
    // x = (0-50)*10/100 = -5
    // y = (0-50)*10/100 = -5
    
    auto cloud = depth_to_cloud(depth, K, 1.0f);
    ASSERT_EQ(cloud.size(), 1);
    EXPECT_NEAR(cloud[0].x, -5.0f, 1e-4);
    EXPECT_NEAR(cloud[0].y, -5.0f, 1e-4);
    EXPECT_NEAR(cloud[0].z, 10.0f, 1e-4);
}
