#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

struct HarrisConfig {
    int blockSize = 2;     // Neighborhood size
    int apertureSize = 3;  // Sobel aperture
    double k = 0.04;       // Harris parameter
};

/**
 * @brief Computes Harris Corner Response map.
 * @param gray Input grayscale image.
 * @param config Harris configuration.
 * @return Response map (CV_32F).
 */
cv::Mat computeHarrisResponse(const cv::Mat& gray, const HarrisConfig& config);

/**
 * @brief Detects corners from response map using NMS.
 * @param response Harris response map.
 * @param threshold Threshold for response.
 * @return List of detected keypoints.
 */
std::vector<cv::Point2f> detectCorners(const cv::Mat& response, float threshold);

} // namespace cv_curriculum
