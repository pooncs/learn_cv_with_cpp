#include "depth_util.hpp"

std::vector<Point3D> depth_to_cloud(const cv::Mat& depth, const cv::Mat& K, float depth_scale) {
    // TODO:
    // 1. Iterate over depth pixels
    // 2. Filter invalid depth
    // 3. Back-project: x = (u-cx)*z/fx, y = (v-cy)*z/fy
    return {};
}
