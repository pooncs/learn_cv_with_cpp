#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

struct OrbConfig {
    int n_points = 256; // Number of pairs (bits)
    int patchSize = 31; // Patch size
};

/**
 * @brief Computes orientation for keypoints.
 * @param image Input grayscale image.
 * @param keypoints Keypoints to process (updates angle).
 */
void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, int patchSize = 31);

/**
 * @brief Extracts ORB-like descriptors.
 * @param image Input grayscale image (should be smoothed).
 * @param keypoints Keypoints.
 * @param config Configuration.
 * @return Descriptors matrix (CV_8U, N x 32).
 */
cv::Mat extractOrbDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const OrbConfig& config);

} // namespace cv_curriculum
