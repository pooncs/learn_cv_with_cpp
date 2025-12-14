#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "fast.hpp"

class FastTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create an image with a clear corner
        // Black background, White square
        img = cv::Mat::zeros(50, 50, CV_8UC1);
        cv::rectangle(img, cv::Point(20, 20), cv::Point(30, 30), cv::Scalar(255), -1);
        // Corners at (20,20), (30,20), (30,30), (20,30)
    }
    cv::Mat img;
};

TEST_F(FastTest, DetectsCorners) {
    cv_curriculum::FastConfig config;
    config.threshold = 50;
    config.N = 9;
    
    auto kps = cv_curriculum::detectFAST(img, config);
    
    // We expect at least 4 corners
    EXPECT_GE(kps.size(), 4);
    
    // Check if one of the corners is near (20, 20)
    bool found = false;
    for (const auto& kp : kps) {
        if (cv::norm(kp.pt - cv::Point2f(20, 20)) < 3.0) {
            found = true;
            break;
        }
    }
    if (!kps.empty()) {
        EXPECT_TRUE(found);
    }
}

TEST_F(FastTest, EmptyImage) {
    cv::Mat empty;
    cv_curriculum::FastConfig config;
    auto kps = cv_curriculum::detectFAST(empty, config);
    EXPECT_TRUE(kps.empty());
}
