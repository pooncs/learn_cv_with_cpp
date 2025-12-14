#pragma once
#include <opencv2/opencv.hpp>

cv::Mat project_points(const cv::Mat& points_3d, const cv::Mat& K);
