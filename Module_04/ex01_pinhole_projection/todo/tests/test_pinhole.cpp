#include <gtest/gtest.h>
#include "pinhole.hpp"

TEST(PinholeTest, CenterPoint) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        100, 0, 50,
        0, 100, 50,
        0, 0, 1);
    
    cv::Mat p3d = (cv::Mat_<double>(1, 3) << 0, 0, 10);
    cv::Mat p2d = project_points(p3d, K);
    
    // u = 100 * 0/10 + 50 = 50
    // v = 100 * 0/10 + 50 = 50
    EXPECT_NEAR(p2d.at<double>(0, 0), 50.0, 1e-5);
    EXPECT_NEAR(p2d.at<double>(0, 1), 50.0, 1e-5);
}
