#pragma once
#include <opencv2/opencv.hpp>

struct StereoMaps {
    cv::Mat map1_left, map2_left;
    cv::Mat map1_right, map2_right;
    cv::Mat Q;
};

StereoMaps compute_stereo_rectification(
    const cv::Mat& K1, const cv::Mat& D1,
    const cv::Mat& K2, const cv::Mat& D2,
    const cv::Mat& R, const cv::Mat& T,
    const cv::Size& img_size);
