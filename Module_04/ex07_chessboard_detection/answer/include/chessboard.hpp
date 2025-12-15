#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Detects chessboard corners and refines them
// Returns true if found, false otherwise
// corners: Output vector of 2D points
bool detect_and_refine_corners(const cv::Mat& img, const cv::Size& board_size, std::vector<cv::Point2f>& corners);
