#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Sorts corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
std::vector<cv::Point2f> sort_corners(const std::vector<cv::Point2f>& pts);

// Rectifies a document defined by 4 corners to a target aspect ratio
// If width/height are 0, they are estimated from the corners
cv::Mat rectify_document(const cv::Mat& src, const std::vector<cv::Point2f>& corners, float aspect_ratio = 0.707f);
