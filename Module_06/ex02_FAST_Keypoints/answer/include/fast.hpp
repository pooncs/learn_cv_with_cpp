#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

struct FastConfig {
    int threshold = 20;    // Intensity difference threshold
    int N = 9;             // Minimum contiguous pixels (usually 9 or 12)
    bool nonmaxSuppression = true;
};

/**
 * @brief Detects keypoints using FAST algorithm.
 * @param gray Input grayscale image.
 * @param config FAST configuration.
 * @return List of keypoints.
 */
std::vector<cv::KeyPoint> detectFAST(const cv::Mat& gray, const FastConfig& config);

} // namespace cv_curriculum
