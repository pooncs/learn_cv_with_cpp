#include <gtest/gtest.h>
#include "homography.hpp"

TEST(HomographyTest, Identity) {
    std::vector<cv::Point2f> pts = {
        {0, 0}, {100, 0}, {100, 100}, {0, 100}
    };
    // Mapping to itself should produce identity matrix
    cv::Mat H = compute_homography(pts, pts);
    
    EXPECT_NEAR(H.at<double>(0,0), 1.0, 1e-5);
    EXPECT_NEAR(H.at<double>(1,1), 1.0, 1e-5);
    EXPECT_NEAR(H.at<double>(2,2), 1.0, 1e-5);
    EXPECT_NEAR(H.at<double>(0,1), 0.0, 1e-5);
}
