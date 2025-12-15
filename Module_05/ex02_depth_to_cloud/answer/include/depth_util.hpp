#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Point3D {
    float x, y, z;
};

// Converts depth image (CV_16U mm or CV_32F m) to point cloud
// depth_scale: scale to convert depth value to meters (e.g. 0.001 for mm->m)
std::vector<Point3D> depth_to_cloud(const cv::Mat& depth, const cv::Mat& K, float depth_scale = 1.0f);
