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
    
    // Extract corresponding points
    std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> dstPoints;
    
    for (const auto& m : matches) {
        srcPoints.push_back(queryPts[m.queryIdx]);
        dstPoints.push_back(trainPts[m.trainIdx]);
    }
    
    // Compute Homography with RANSAC
    std::vector<uchar> inliersMask;
    result.H = cv::findHomography(srcPoints, dstPoints, cv::RANSAC, ransacReprojThreshold, inliersMask);
    
    // Filter matches
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inliersMask[i]) {
            result.inlierMatches.push_back(matches[i]);
        }
    }
    
    return result;
}

} // namespace cv_curriculum
