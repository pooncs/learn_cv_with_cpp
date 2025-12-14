#pragma once
#include <opencv2/opencv.hpp>

// Returns the normalized ray direction (x, y, 1) for pixel (u, v)
cv::Point3d pixel_to_ray(double u, double v, const cv::Mat& K);

// Returns 3D point at depth Z
cv::Point3d reconstruct_point(double u, double v, double Z, const cv::Mat& K);
