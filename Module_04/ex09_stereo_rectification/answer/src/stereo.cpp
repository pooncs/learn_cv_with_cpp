#include "stereo.hpp"

StereoMaps compute_stereo_rectification(
    const cv::Mat& K1, const cv::Mat& D1,
    const cv::Mat& K2, const cv::Mat& D2,
    const cv::Mat& R, const cv::Mat& T,
    const cv::Size& img_size) 
{
    StereoMaps maps;
    cv::Mat R1, R2, P1, P2;
    
    cv::stereoRectify(K1, D1, K2, D2, img_size, R, T, R1, R2, P1, P2, maps.Q, 
        cv::CALIB_ZERO_DISPARITY, 0, img_size);

    cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_32FC1, maps.map1_left, maps.map2_left);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_32FC1, maps.map1_right, maps.map2_right);

    return maps;
}
