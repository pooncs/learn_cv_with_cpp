#include "undistort.hpp"

std::pair<cv::Mat, cv::Mat> compute_undistortion_maps(
    const cv::Mat& K, const cv::Mat& dist_coeffs, const cv::Size& size) 
{
    // TODO:
    // 1. Create map_x, map_y (CV_32FC1)
    // 2. Loop over pixels (u, v)
    // 3. Normalize -> Distort -> Project
    // 4. Store u_src, v_src in maps
    return {cv::Mat(), cv::Mat()};
}
