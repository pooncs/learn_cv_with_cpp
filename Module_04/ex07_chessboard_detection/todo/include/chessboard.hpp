#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

bool detect_and_refine_corners(const cv::Mat& img, const cv::Size& board_size, std::vector<cv::Point2f>& corners);
