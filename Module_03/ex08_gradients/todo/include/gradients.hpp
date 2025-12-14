#pragma once
#include <opencv2/opencv.hpp>
#include <utility>

// Computes Sobel X and Y derivatives (CV_32F)
// Returns pair {Gx, Gy}
std::pair<cv::Mat, cv::Mat> compute_sobel(const cv::Mat& src);

// Computes magnitude and angle (degrees) from gradients
// Returns pair {Magnitude, Angle}
std::pair<cv::Mat, cv::Mat> compute_magnitude_angle(const cv::Mat& Gx, const cv::Mat& Gy);
