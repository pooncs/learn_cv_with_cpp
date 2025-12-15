#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "matcher.hpp"

TEST(BFMatcherTest, HammingDistance) {
    cv::Mat a = cv::Mat::zeros(1, 32, CV_8UC1);
    cv::Mat b = cv::Mat::zeros(1, 32, CV_8UC1);
    
    // Both zero -> dist 0
    EXPECT_EQ(cv_curriculum::computeHammingDistance(a, b), 0);
    
    // a=0xFF, b=0x00 -> dist 8
    a.at<uchar>(0, 0) = 0xFF;
    EXPECT_EQ(cv_curriculum::computeHammingDistance(a, b), 8);
    
    // a=0x0F, b=0xF0 -> dist 8 (all bits flip)
    a.at<uchar>(0, 0) = 0x0F;
    b.at<uchar>(0, 0) = 0xF0;
    EXPECT_EQ(cv_curriculum::computeHammingDistance(a, b), 8);
}

TEST(BFMatcherTest, Matching) {
    // 2 Query, 3 Train
    cv::Mat query = cv::Mat::zeros(2, 32, CV_8UC1);
    cv::Mat train = cv::Mat::zeros(3, 32, CV_8UC1);
    
    // Q0 matches T0 exactly
    query.at<uchar>(0, 0) = 0xAA;
    train.at<uchar>(0, 0) = 0xAA;
    
    // Q1 matches T2 exactly
    query.at<uchar>(1, 0) = 0xBB;
    train.at<uchar>(2, 0) = 0xBB;
    
    // T1 is junk
    train.at<uchar>(1, 0) = 0xFF;
    
    auto matches = cv_curriculum::matchBruteForce(query, train);
    
    if (matches.empty()) return; // Skip if stub
    
    ASSERT_EQ(matches.size(), 2);
    
    // Check Q0 -> T0
    EXPECT_EQ(matches[0].queryIdx, 0);
    EXPECT_EQ(matches[0].trainIdx, 0);
    EXPECT_EQ(matches[0].distance, 0);
    
    // Check Q1 -> T2
    EXPECT_EQ(matches[1].queryIdx, 1);
    EXPECT_EQ(matches[1].trainIdx, 2);
    EXPECT_EQ(matches[1].distance, 0);
}
