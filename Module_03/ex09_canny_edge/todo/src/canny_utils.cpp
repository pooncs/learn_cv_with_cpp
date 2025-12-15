#include "canny_utils.hpp"

cv::Mat non_max_suppression(const cv::Mat& mag, const cv::Mat& angle) {
    // TODO: Implement NMS
    // 1. Loop through pixels (skip borders)
    // 2. Normalize angle to 0, 45, 90, 135
    // 3. Compare mag(i,j) with neighbors in gradient direction
    // 4. If mag(i,j) is max, keep it, else set to 0
    return cv::Mat();
}
