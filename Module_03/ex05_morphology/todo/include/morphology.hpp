#pragma once
#include <opencv2/opencv.hpp>

// Binary Dilation (255=foreground, 0=background)
// 3x3 structuring element
cv::Mat my_dilate(const cv::Mat& src);

// Binary Erosion
cv::Mat my_erode(const cv::Mat& src);
