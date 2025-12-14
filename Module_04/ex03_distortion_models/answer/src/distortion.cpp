#include "distortion.hpp"

cv::Point2d distort_point(double x, double y, double k1, double k2, double p1, double p2) {
    double r2 = x*x + y*y;
    double r4 = r2 * r2;
    
    double radial = 1.0 + k1 * r2 + k2 * r4;
    
    double x_rad = x * radial;
    double y_rad = y * radial;
    
    double x_tan = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
    double y_tan = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
    
    return cv::Point2d(x_rad + x_tan, y_rad + y_tan);
}

cv::Point2d project_distorted(const cv::Point3d& P, const cv::Mat& K, const cv::Mat& dist_coeffs) {
    double x = P.x / P.z;
    double y = P.y / P.z;
    
    double k1 = dist_coeffs.at<double>(0);
    double k2 = dist_coeffs.at<double>(1);
    double p1 = dist_coeffs.at<double>(2);
    double p2 = dist_coeffs.at<double>(3);
    
    cv::Point2d p_dist = distort_point(x, y, k1, k2, p1, p2);
    
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    
    return cv::Point2d(fx * p_dist.x + cx, fy * p_dist.y + cy);
}
