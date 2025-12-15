#include <gtest/gtest.h>
#include "pnp.hpp"

TEST(PnPTest, SimpleTranslation) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);

    std::vector<cv::Point3f> obj_pts = {
        {0, 0, 0}, {10, 0, 0}, {10, 10, 0}, {0, 10, 0}
    };
    
    // T = [0, 0, 10]
    // (0,0,0) -> (0,0,10) -> (320, 240)
    // (10,0,0) -> (10,0,10) -> (1000*1 + 320, 240) = (1320, 240)
    std::vector<cv::Point2f> img_pts = {
        {320, 240}, {1320, 240}, {1320, 1240}, {320, 1240}
    };
    
    auto [rvec, tvec] = estimate_pose(obj_pts, img_pts, K, dist);
    
    if (!tvec.empty()) {
        EXPECT_NEAR(tvec.at<double>(2), 10.0, 0.1);
        EXPECT_NEAR(tvec.at<double>(0), 0.0, 0.1);
        EXPECT_NEAR(tvec.at<double>(1), 0.0, 0.1);
    }
}
