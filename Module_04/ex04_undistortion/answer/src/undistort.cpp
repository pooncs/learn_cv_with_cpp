#include "undistort.hpp"

std::pair<cv::Mat, cv::Mat> compute_undistortion_maps(
    const cv::Mat& K, const cv::Mat& dist_coeffs, const cv::Size& size) 
{
    cv::Mat map_x(size, CV_32FC1);
    cv::Mat map_y(size, CV_32FC1);

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    double k1 = dist_coeffs.at<double>(0);
    double k2 = dist_coeffs.at<double>(1);
    double p1 = dist_coeffs.at<double>(2);
    double p2 = dist_coeffs.at<double>(3);
    // Assuming k3 = 0 or handles up to 5 coeffs

    for (int v = 0; v < size.height; ++v) {
        for (int u = 0; u < size.width; ++u) {
            // 1. Unproject (Normalize)
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;

            // 2. Distort
            double r2 = x*x + y*y;
            double r4 = r2 * r2;
            double radial = 1.0 + k1 * r2 + k2 * r4;
            
            double x_rad = x * radial;
            double y_rad = y * radial;
            
            double x_tan = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
            double y_tan = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

            double x_dist = x_rad + x_tan;
            double y_dist = y_rad + y_tan;

            // 3. Project back to pixel coords
            double u_src = fx * x_dist + cx;
            double v_src = fy * y_dist + cy;

            map_x.at<float>(v, u) = static_cast<float>(u_src);
            map_y.at<float>(v, u) = static_cast<float>(v_src);
        }
    }

    return {map_x, map_y};
}
