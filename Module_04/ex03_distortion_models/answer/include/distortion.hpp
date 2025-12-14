#pragma once
#include <opencv2/opencv.hpp>

// Distorts normalized point (x, y) using k1, k2, p1, p2
cv::Point2d distort_point(double x, double y, double k1, double k2, double p1, double p2);

// Projects and distorts
cv::Point2d project_distorted(const cv::Point3d& P, const cv::Mat& K, const cv::Mat& dist_coeffs);
