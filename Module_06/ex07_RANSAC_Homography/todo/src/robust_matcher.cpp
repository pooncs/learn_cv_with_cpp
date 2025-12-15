#include "robust_matcher.hpp"

namespace cv_curriculum {

RobustMatchResult computeRobustHomography(
    const std::vector<cv::Point2f>& queryPts,
    const std::vector<cv::Point2f>& trainPts,
    const std::vector<cv::DMatch>& matches,
    double ransacReprojThreshold)
{
    RobustMatchResult result;
    if (matches.size() < 4) return result;
    
    // TODO: Implement RANSAC Homography
    // 1. Extract matched points from queryPts and trainPts using matches indices.
    // 2. Use cv::findHomography with cv::RANSAC.
    // 3. Use the returned mask to populate result.inlierMatches.
    
    return result;
}

} // namespace cv_curriculum
