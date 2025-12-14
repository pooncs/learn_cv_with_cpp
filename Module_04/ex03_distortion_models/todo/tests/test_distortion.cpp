#include <gtest/gtest.h>
#include "distortion.hpp"

TEST(DistortionTest, NoDistortion) {
    // k1=k2=p1=p2=0
    cv::Point2d p = distort_point(0.1, 0.1, 0, 0, 0, 0);
    EXPECT_DOUBLE_EQ(p.x, 0.1);
    EXPECT_DOUBLE_EQ(p.y, 0.1);
}
