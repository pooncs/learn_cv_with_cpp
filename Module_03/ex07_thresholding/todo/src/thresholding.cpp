#include "thresholding.hpp"

cv::Mat my_threshold(const cv::Mat& src, int thresh) {
    // TODO: Implement global thresholding
    return cv::Mat();
}

cv::Mat my_adaptive_threshold(const cv::Mat& src, int blockSize, int C) {
    // TODO: Implement adaptive thresholding
    // 1. Calculate mean (use cv::blur)
    // 2. Threshold: pixel > mean - C ? 255 : 0
    return cv::Mat();
}
