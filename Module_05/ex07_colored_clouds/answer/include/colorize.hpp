#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

struct PointRGB {
    float x, y, z;
    uint8_t r, g, b;
};

// Colors a point cloud using an RGB image and intrinsics
std::vector<PointRGB> colorize_cloud(const std::vector<cv::Point3f>& cloud, const cv::Mat& rgb_img, const cv::Mat& K);
