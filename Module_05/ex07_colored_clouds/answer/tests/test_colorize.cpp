#include <gtest/gtest.h>
#include "colorize.hpp"

TEST(ColorizeTest, CenterRed) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        100, 0, 50,
        0, 100, 50,
        0, 0, 1);
    
    cv::Mat img = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 255)); // Red
    
    // Point on Z axis -> Center of image
    std::vector<cv::Point3f> cloud = {{0, 0, 10}};
    
    auto colored = colorize_cloud(cloud, img, K);
    
    ASSERT_EQ(colored.size(), 1);
    EXPECT_EQ(colored[0].r, 255);
    EXPECT_EQ(colored[0].g, 0);
    EXPECT_EQ(colored[0].b, 0);
}
