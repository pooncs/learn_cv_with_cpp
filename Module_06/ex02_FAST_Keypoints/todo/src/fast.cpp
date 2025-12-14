#include "fast.hpp"

namespace cv_curriculum {

std::vector<cv::KeyPoint> detectFAST(const cv::Mat& gray, const FastConfig& config) {
    std::vector<cv::KeyPoint> keypoints;
    if (gray.empty()) return keypoints;

    // TODO: Implement FAST
    // 1. Iterate over all pixels (excluding border radius 3)
    // 2. For each pixel p, check circle of 16 pixels
    // 3. Check for N contiguous pixels > p + t or < p - t
    // 4. Compute score (optional but good for NMS)
    // 5. Perform Non-Maximum Suppression if enabled
    
    return keypoints;
}

} // namespace cv_curriculum
