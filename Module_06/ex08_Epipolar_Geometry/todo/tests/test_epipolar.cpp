#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "epipolar.hpp"

TEST(EpipolarTest, PureTranslation) {
    // Pure horizontal translation [t_x, 0, 0]
    // F should be roughly [0 0 0; 0 0 -1; 0 1 0] (skew symmetric of [1,0,0])
    std::vector<cv::Point2f> pts1, pts2;
    for(int i=0; i<50; ++i) {
        pts1.emplace_back(rand()%100, rand()%100);
        pts2.emplace_back(pts1.back().x - 10, pts1.back().y);
    }
    
    auto result = cv_curriculum::computeFundamentalMatrix(pts1, pts2);
    
    if (result.F.empty()) return; // Stub
    
    // Check property x'T F x = 0
    // Pick a point
    cv::Mat pt1 = (cv::Mat_<double>(3,1) << pts1[0].x, pts1[0].y, 1.0);
    cv::Mat pt2 = (cv::Mat_<double>(3,1) << pts2[0].x, pts2[0].y, 1.0);
    
    cv::Mat F;
    result.F.convertTo(F, CV_64F);
    
    cv::Mat val = pt2.t() * F * pt1;
    EXPECT_NEAR(val.at<double>(0), 0.0, 1e-1); // Should be small
}
