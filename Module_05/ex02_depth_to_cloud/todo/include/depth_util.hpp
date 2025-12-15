#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Point3D {
    float x, y, z;
};

std::vector<Point3D> depth_to_cloud(const cv::Mat& depth, const cv::Mat& K, float depth_scale = 1.0f);
