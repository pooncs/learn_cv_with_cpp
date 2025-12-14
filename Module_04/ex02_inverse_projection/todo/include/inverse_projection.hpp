#pragma once
#include <opencv2/opencv.hpp>

cv::Point3d pixel_to_ray(double u, double v, const cv::Mat& K);
cv::Point3d reconstruct_point(double u, double v, double Z, const cv::Mat& K);
