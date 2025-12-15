#include "warp.hpp"

cv::Mat rectify_document(const cv::Mat& src, const std::vector<cv::Point2f>& corners, float aspect_ratio) {
    // TODO:
    // 1. Sort corners (TL, TR, BR, BL)
    // 2. Compute width/height
    // 3. Define dst points
    // 4. Compute Perspective Transform
    // 5. Warp
    return cv::Mat();
}
