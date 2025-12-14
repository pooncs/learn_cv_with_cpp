#pragma once
#include <opencv2/opencv.hpp>

cv::Mat my_threshold(const cv::Mat& src, int thresh);
cv::Mat my_adaptive_threshold(const cv::Mat& src, int blockSize, int C);
