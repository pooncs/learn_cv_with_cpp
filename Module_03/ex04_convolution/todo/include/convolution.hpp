#pragma once
#include <opencv2/opencv.hpp>

// Performs 3x3 convolution (correlation) with zero padding
// src: CV_8UC1
// kernel: CV_32F, 3x3
// Returns: CV_8UC1
cv::Mat convolve(const cv::Mat& src, const cv::Mat& kernel);
