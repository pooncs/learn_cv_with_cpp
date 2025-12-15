#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "flann_matcher.hpp"

TEST(FlannTest, Creation) {
    auto matcher = cv_curriculum::createFlannLshMatcher();
    ASSERT_FALSE(matcher.empty());
    
    // Can't easily inspect internal params, but can check if it works with binary descriptors
    cv::Mat query = cv::Mat::zeros(5, 32, CV_8UC1);
    cv::Mat train = cv::Mat::zeros(5, 32, CV_8UC1);
    
    // Should NOT throw exception
    std::vector<std::vector<cv::DMatch>> matches;
    EXPECT_NO_THROW(matcher->knnMatch(query, train, matches, 2));
}

TEST(FlannTest, Matching) {
    auto matcher = cv_curriculum::createFlannLshMatcher();
    if (matcher.empty()) return;
    
    // Exact match test
    cv::Mat query = cv::Mat::zeros(1, 32, CV_8UC1);
    query.at<uchar>(0, 0) = 0xFF;
    
    cv::Mat train = cv::Mat::zeros(2, 32, CV_8UC1);
    train.at<uchar>(0, 0) = 0xFF; // Match
    train.at<uchar>(1, 0) = 0x00; // No Match
    
    auto good = cv_curriculum::matchFlann(matcher, query, train, 0.9f);
    
    ASSERT_EQ(good.size(), 1);
    EXPECT_EQ(good[0].trainIdx, 0);
}
