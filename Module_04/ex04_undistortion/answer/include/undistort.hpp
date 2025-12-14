#pragma once
#include <opencv2/opencv.hpp>
#include <utility>

// Computes map_x and map_y for cv::remap
// Returns {map_x, map_y} (CV_32FC1)
std::pair<cv::Mat, cv::Mat> compute_undistortion_maps(
    const cv::Mat& K, const cv::Mat& dist_coeffs, const cv::Size& size);
