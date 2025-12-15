#include <gtest/gtest.h>
#include "stereo.hpp"

TEST(StereoTest, IdentityRectification) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat T = (cv::Mat_<double>(3, 1) << 1, 0, 0); // Horizontal shift
    cv::Size size(100, 100);
    
    StereoMaps maps = compute_stereo_rectification(K, D, K, D, R, T, size);
    
    EXPECT_FALSE(maps.map1_left.empty());
    EXPECT_FALSE(maps.map2_left.empty());
    EXPECT_FALSE(maps.Q.empty());
}
