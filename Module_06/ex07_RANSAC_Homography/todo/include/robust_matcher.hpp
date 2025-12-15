#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

struct RobustMatchResult {
    cv::Mat H; // Homography matrix (3x3)
    std::vector<cv::DMatch> inlierMatches;
};

/**
 * @brief Computes homography using RANSAC and filters outliers.
 * @param queryPts Points from query image.
 * @param trainPts Points from train image.
 * @param matches Initial matches (indices align with pts if we passed raw points, but usually we pass keypoints).
 *                Here we assume caller extracted Point2f corresponding to matches.
 * @param ransacReprojThreshold Max error in pixels.
 * @return Result containing H and the list of inliers.
 */
RobustMatchResult computeRobustHomography(
    const std::vector<cv::Point2f>& queryPts,
    const std::vector<cv::Point2f>& trainPts,
    const std::vector<cv::DMatch>& matches,
    double ransacReprojThreshold = 3.0);

} // namespace cv_curriculum
