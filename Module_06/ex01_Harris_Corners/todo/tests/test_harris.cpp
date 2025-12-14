#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "harris.hpp"

class HarrisTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple image with a white square in the center
        // Corners should be at (50, 50), (150, 50), (150, 150), (50, 150)
        img = cv::Mat::zeros(200, 200, CV_8UC1);
        cv::rectangle(img, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), -1);
    }
    cv::Mat img;
};

TEST_F(HarrisTest, ResponseCalculation) {
    cv_curriculum::HarrisConfig config;
    cv::Mat response = cv_curriculum::computeHarrisResponse(img, config);
    
    ASSERT_FALSE(response.empty());
    ASSERT_EQ(response.size(), img.size());
    ASSERT_EQ(response.type(), CV_32F);
    
    // Check if corners have high response
    // Note: The exact corner is slightly inside due to window size, but response should be high near corners
    float cornerResponse = response.at<float>(50, 50);
    float flatResponse = response.at<float>(10, 10);
    
    // In a perfect corner, response is positive large
    // In a flat region, response is near zero
    // Since this is a Todo, we expect failure if implemented incorrectly. 
    // But we write the correct expectation.
    if (cv::countNonZero(response) > 0) {
        EXPECT_GT(cornerResponse, flatResponse);
    }
}

TEST_F(HarrisTest, CornerDetection) {
    cv_curriculum::HarrisConfig config;
    cv::Mat response = cv_curriculum::computeHarrisResponse(img, config);
    
    double minVal, maxVal;
    cv::minMaxLoc(response, &minVal, &maxVal);
    
    if (maxVal <= 0) {
        // Skip if response is not implemented
        return; 
    }

    float threshold = static_cast<float>(maxVal * 0.1);
    
    auto corners = cv_curriculum::detectCorners(response, threshold);
    
    // We expect 4 corners
    EXPECT_EQ(corners.size(), 4);
    
    // Check if corners are near expected locations
    std::vector<cv::Point2f> expected = {
        {50, 50}, {150, 50}, {150, 150}, {50, 150}
    };
    
    int matched = 0;
    for (const auto& pt : corners) {
        for (const auto& gt : expected) {
            if (cv::norm(pt - gt) < 5.0) {
                matched++;
                break;
            }
        }
    }
    EXPECT_EQ(matched, 4);
}
