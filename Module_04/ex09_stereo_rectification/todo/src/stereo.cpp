#include "stereo.hpp"

StereoMaps compute_stereo_rectification(
    const cv::Mat& K1, const cv::Mat& D1,
    const cv::Mat& K2, const cv::Mat& D2,
    const cv::Mat& R, const cv::Mat& T,
    const cv::Size& img_size) 
{
    // TODO:
    // 1. cv::stereoRectify
    // 2. cv::initUndistortRectifyMap (Left)
    // 3. cv::initUndistortRectifyMap (Right)
    return StereoMaps();
}
