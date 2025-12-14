#pragma once
#include <opencv2/opencv.hpp>
#include <utility>

std::pair<cv::Mat, cv::Mat> compute_undistortion_maps(
    const cv::Mat& K, const cv::Mat& dist_coeffs, const cv::Size& size);
