#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

/**
 * @brief Finds K nearest neighbors using Brute Force.
 * @param queryDescriptors Query descriptors.
 * @param trainDescriptors Train descriptors.
 * @param k Number of neighbors (usually 2).
 * @return Vector of vectors (matches per query).
 */
std::vector<std::vector<cv::DMatch>> matchKnnBruteForce(
    const cv::Mat& queryDescriptors, 
    const cv::Mat& trainDescriptors, 
    int k = 2);

/**
 * @brief Filters matches using Lowe's Ratio Test.
 * @param knnMatches Input matches (must have size >= 2 per query).
 * @param ratio Threshold ratio (e.g., 0.75).
 * @return Filtered good matches.
 */
std::vector<cv::DMatch> filterRatioTest(
    const std::vector<std::vector<cv::DMatch>>& knnMatches, 
    float ratio = 0.75f);

} // namespace cv_curriculum
