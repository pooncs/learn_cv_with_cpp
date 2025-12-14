#include <gtest/gtest.h>
#include "inverse_projection.hpp"

TEST(InverseProjTest, CenterPixel) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);
    
    cv::Point3d ray = pixel_to_ray(320, 240, K);
    EXPECT_NEAR(ray.x, 0.0, 1e-5);
    EXPECT_NEAR(ray.y, 0.0, 1e-5);
    EXPECT_NEAR(ray.z, 1.0, 1e-5);
}
