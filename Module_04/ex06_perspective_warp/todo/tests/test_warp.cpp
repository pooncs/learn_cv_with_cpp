#include <gtest/gtest.h>
#include "warp.hpp"

TEST(WarpTest, SimpleRect) {
    // 100x100 square at (0,0)
    std::vector<cv::Point2f> corners = {
        {0, 0}, {100, 0}, {100, 100}, {0, 100}
    };
    
    cv::Mat src = cv::Mat::zeros(200, 200, CV_8UC1);
    
    // Rectify with AR=1.0 should keep it roughly 100x100
    cv::Mat res = rectify_document(src, corners, 1.0f);
    
    // The implementation might output exactly max width/height
    if (!res.empty()) {
        EXPECT_NEAR(res.cols, 100, 2);
        EXPECT_NEAR(res.rows, 100, 2);
    }
}
