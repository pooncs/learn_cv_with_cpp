#pragma once
#include <opencv2/opencv.hpp>

// Performs Non-Maximum Suppression
// mag: Gradient Magnitude (CV_32F)
// angle: Gradient Angle in degrees (CV_32F)
// Returns: Thinned edge map (CV_8UC1 or CV_32F)
cv::Mat non_max_suppression(const cv::Mat& mag, const cv::Mat& angle);
