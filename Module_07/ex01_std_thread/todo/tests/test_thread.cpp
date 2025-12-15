#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "parallel_process.hpp"

TEST(ThreadTest, Invert) {
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC1);
    img.setTo(100);
    
    cv_curriculum::invertImageParallel(img, 4);
    
    // 255 - 100 = 155
    EXPECT_EQ(img.at<uchar>(0, 0), 155);
    EXPECT_EQ(img.at<uchar>(50, 50), 155);
    EXPECT_EQ(img.at<uchar>(99, 99), 155);
}
