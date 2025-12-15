#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

/**
 * @brief Computes Hamming distance between two binary descriptors.
 * @param a First descriptor (row).
 * @param b Second descriptor (row).
 * @return Hamming distance (number of differing bits).
 */
int computeHammingDistance(const cv::Mat& a, const cv::Mat& b);

/**
 * @brief Matches query descriptors to train descriptors using Brute Force.
 * @param queryDescriptors Descriptors from query image (M x 32).
 * @param trainDescriptors Descriptors from train image (N x 32).
 * @return List of DMatch objects (M matches).
 */
std::vector<cv::DMatch> matchBruteForce(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors);

} // namespace cv_curriculum
