#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Projects 3D points to 2D image plane using pinhole model.
 * 
 * @param points_3d Nx3 matrix of 3D points (CV_32F or CV_64F)
 * @param K 3x3 Intrinsic matrix
 * @return Nx2 matrix of 2D points
 */
cv::Mat project_points(const cv::Mat& points_3d, const cv::Mat& K);
