#include "pinhole.hpp"

cv::Mat project_points(const cv::Mat& points_3d, const cv::Mat& K) {
    // TODO: Implement pinhole projection
    // 1. Extract fx, fy, cx, cy from K
    // 2. Iterate over points
    // 3. Compute u = fx * X/Z + cx
    // 4. Compute v = fy * Y/Z + cy
    return cv::Mat();
}
