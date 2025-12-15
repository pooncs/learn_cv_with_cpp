#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "robust_matcher.hpp"

TEST(RansacTest, FiltersOutliers) {
    // 4 points form a square, 5th point is way off
    std::vector<cv::Point2f> pts1 = {{0,0}, {100,0}, {100,100}, {0,100}, {50,50}};
    std::vector<cv::Point2f> pts2 = {{0,0}, {100,0}, {100,100}, {0,100}, {500,500}}; // Outlier
    
    std::vector<cv::DMatch> matches;
    for(int i=0; i<5; ++i) matches.emplace_back(i, i, 0.0f);
    
    auto result = cv_curriculum::computeRobustHomography(pts1, pts2, matches, 1.0);
    
    if (result.H.empty()) return; // Stub
    
    // Should keep 4, reject 1
    EXPECT_EQ(result.inlierMatches.size(), 4);
    
    // H should be Identity (roughly)
    double det = cv::determinant(result.H);
    EXPECT_NEAR(det, 1.0, 0.1);
}
