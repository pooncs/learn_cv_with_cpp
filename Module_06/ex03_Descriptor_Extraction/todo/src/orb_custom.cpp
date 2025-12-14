#include "orb_custom.hpp"

namespace cv_curriculum {

void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, int patchSize) {
    // TODO: Implement Intensity Centroid
    // 1. Iterate over keypoints
    // 2. For each keypoint, iterate over patch
    // 3. Compute moments m01, m10
    // 4. Compute angle = atan2(m01, m10) * 180 / PI
    // 5. Store in keypoint.angle
}

cv::Mat extractOrbDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const OrbConfig& config) {
    cv::Mat descriptors;
    
    // TODO: Implement Rotated BRIEF
    // 1. Define pattern (random pairs)
    // 2. Smooth image
    // 3. For each keypoint:
    //    a. Get angle, compute sin/cos
    //    b. For each pair in pattern:
    //       i. Rotate points by angle
    //       ii. Compare intensity: I(p1) < I(p2) ? 1 : 0
    //       iii. Pack into descriptor row
    
    return descriptors;
}

} // namespace cv_curriculum
