#include "gradients.hpp"

std::pair<cv::Mat, cv::Mat> compute_sobel(const cv::Mat& src) {
    // TODO:
    // 1. Convert to Gray if needed
    // 2. Compute Sobel X (dx=1, dy=0)
    // 3. Compute Sobel Y (dx=0, dy=1)
    // Return {Gx, Gy} in CV_32F
    return {cv::Mat(), cv::Mat()};
}

std::pair<cv::Mat, cv::Mat> compute_magnitude_angle(const cv::Mat& Gx, const cv::Mat& Gy) {
    // TODO:
    // 1. Compute Magnitude sqrt(Gx^2 + Gy^2)
    // 2. Compute Angle atan2(Gy, Gx) * 180 / PI
    return {cv::Mat(), cv::Mat()};
}
