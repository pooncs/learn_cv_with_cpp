#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

/**
 * @brief Triangulates 3D points from stereo matches.
 * @param P1 Projection matrix of camera 1 (3x4).
 * @param P2 Projection matrix of camera 2 (3x4).
 * @param pts1 Points in image 1.
 * @param pts2 Points in image 2.
 * @return Vector of 3D points.
 */
std::vector<cv::Point3f> triangulateStereo(
    const cv::Mat& P1,
    const cv::Mat& P2,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2);

} // namespace cv_curriculum
