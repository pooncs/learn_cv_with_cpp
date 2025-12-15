#include "pnp.hpp"

std::pair<cv::Mat, cv::Mat> estimate_pose(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& K,
    const cv::Mat& dist_coeffs) 
{
    // TODO: cv::solvePnP
    return {cv::Mat(), cv::Mat()};
}

void draw_axes(cv::Mat& img, const cv::Mat& K, const cv::Mat& dist_coeffs, const cv::Mat& rvec, const cv::Mat& tvec, float length) {
    // TODO: 
    // 1. Define axis points (0,0,0), (L,0,0), etc.
    // 2. Project points
    // 3. Draw lines
}
