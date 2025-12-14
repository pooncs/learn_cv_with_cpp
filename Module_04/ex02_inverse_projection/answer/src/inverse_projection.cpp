#include "inverse_projection.hpp"

cv::Point3d pixel_to_ray(double u, double v, const cv::Mat& K) {
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    double x = (u - cx) / fx;
    double y = (v - cy) / fy;
    return cv::Point3d(x, y, 1.0);
}

cv::Point3d reconstruct_point(double u, double v, double Z, const cv::Mat& K) {
    cv::Point3d ray = pixel_to_ray(u, v, K);
    return ray * Z;
}
