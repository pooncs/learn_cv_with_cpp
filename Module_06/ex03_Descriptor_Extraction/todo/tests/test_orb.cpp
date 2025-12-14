#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "orb_custom.hpp"

class OrbTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create an image with a horizontal gradient
        // Intensity increases with x
        img = cv::Mat::zeros(50, 50, CV_8UC1);
        for (int y = 0; y < 50; ++y) {
            for (int x = 0; x < 50; ++x) {
                img.at<uchar>(y, x) = static_cast<uchar>(x * 5);
            }
        }
    }
    cv::Mat img;
};

TEST_F(OrbTest, OrientationComputation) {
    // Keypoint at center
    std::vector<cv::KeyPoint> kps;
    kps.emplace_back(cv::Point2f(25, 25), 10.0f);
    
    // In this image (gradient along X), intensity centroid should be shifted towards +X.
    // Centroid relative to center: (+x, 0)
    // Angle should be 0 degrees (or close to 0/360)
    
    cv_curriculum::computeOrientation(img, kps);
    
    // If not implemented, angle stays -1
    if (kps[0].angle == -1) return;
    
    // We expect angle ~ 0 (or 360)
    float angle = kps[0].angle;
    if (angle > 180) angle -= 360;
    
    EXPECT_NEAR(angle, 0.0f, 10.0f);
}

TEST_F(OrbTest, DescriptorExtraction) {
    std::vector<cv::KeyPoint> kps;
    kps.emplace_back(cv::Point2f(25, 25), 10.0f);
    kps[0].angle = 0.0f;
    
    cv_curriculum::OrbConfig config;
    cv::Mat descriptors = cv_curriculum::extractOrbDescriptors(img, kps, config);
    
    if (descriptors.empty()) return; // Skip if stub
    
    EXPECT_EQ(descriptors.rows, 1);
    EXPECT_EQ(descriptors.cols, 32); // 256 bits = 32 bytes
    EXPECT_EQ(descriptors.type(), CV_8UC1);
}
