#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat rectify_document(const cv::Mat& src, const std::vector<cv::Point2f>& corners, float aspect_ratio = 0.707f);
