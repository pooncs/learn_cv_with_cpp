#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

struct EpipolarResult {
    cv::Mat F; // Fundamental Matrix
    std::vector<cv::Point2f> inliers1;
    std::vector<cv::Point2f> inliers2;
};

/**
 * @brief Computes Fundamental Matrix using RANSAC.
 * @param pts1 Points in image 1.
 * @param pts2 Points in image 2.
 * @return Result with F and inliers.
 */
EpipolarResult computeFundamentalMatrix(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2);

/**
 * @brief Draws epipolar lines on the image.
 * @param img Image to draw on.
 * @param lines Lines (a, b, c) computed by computeCorrespondEpilines.
 * @param pts Points associated with lines (optional, for drawing dots).
 */
void drawEpipolarLines(
    cv::Mat& img, 
    const std::vector<cv::Vec3f>& lines, 
    const std::vector<cv::Point2f>& pts);

} // namespace cv_curriculum
