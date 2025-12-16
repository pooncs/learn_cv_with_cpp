#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// Calculate Intersection over Union
float iou(const cv::Rect& a, const cv::Rect& b);

// Perform Non-Maximum Suppression
std::vector<Detection> nms(std::vector<Detection>& dets, float threshold);
