#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

// Returns {rvec, tvec}
std::pair<cv::Mat, cv::Mat> estimate_pose(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& K,
    const cv::Mat& dist_coeffs);

void draw_axes(cv::Mat& img, const cv::Mat& K, const cv::Mat& dist_coeffs, const cv::Mat& rvec, const cv::Mat& tvec, float length);
