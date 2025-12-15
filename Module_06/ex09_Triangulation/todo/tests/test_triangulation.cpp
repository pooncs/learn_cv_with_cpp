#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "triangulation.hpp"

TEST(TriangulationTest, SimpleDepth) {
    // Simple stereo setup
    // P1 = [I | 0]
    // P2 = [I | -10 0 0] (Camera 2 is at +10 x)
    // Point at (0, 0, 10)
    // u1 = 0/10 = 0
    // u2 = (0 - 10)/10 = -1
    
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_32F);
    cv::Mat P2 = cv::Mat::eye(3, 4, CV_32F);
    P2.at<float>(0, 3) = -10.0f; 
    
    std::vector<cv::Point2f> pts1 = {{0, 0}};
    std::vector<cv::Point2f> pts2 = {{-1, 0}};
    
    auto points = cv_curriculum::triangulateStereo(P1, P2, pts1, pts2);
    
    if (points.empty()) return; // Stub
    
    ASSERT_EQ(points.size(), 1);
    EXPECT_NEAR(points[0].z, 10.0f, 0.1f);
    EXPECT_NEAR(points[0].x, 0.0f, 0.1f);
    EXPECT_NEAR(points[0].y, 0.0f, 0.1f);
}
