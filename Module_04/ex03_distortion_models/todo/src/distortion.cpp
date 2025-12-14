#include "distortion.hpp"

cv::Point2d distort_point(double x, double y, double k1, double k2, double p1, double p2) {
    // TODO: Implement radial and tangential distortion
    return cv::Point2d(x, y);
}

cv::Point2d project_distorted(const cv::Point3d& P, const cv::Mat& K, const cv::Mat& dist_coeffs) {
    // TODO: Project P to normalized coords, apply distortion, then apply K
    return cv::Point2d(0, 0);
}
