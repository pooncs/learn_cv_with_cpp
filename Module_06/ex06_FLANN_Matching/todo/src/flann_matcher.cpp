#include "flann_matcher.hpp"

namespace cv_curriculum {

cv::Ptr<cv::DescriptorMatcher> createFlannLshMatcher() {
    // TODO: Create FlannBasedMatcher with LshIndexParams
    // LshIndexParams(int table_number, int key_size, int multi_probe_level)
    // Recommended: 12, 20, 2
    return nullptr;
}

std::vector<cv::DMatch> matchFlann(
    cv::Ptr<cv::DescriptorMatcher> matcher,
    const cv::Mat& query,
    const cv::Mat& train,
    float ratio)
{
    std::vector<cv::DMatch> goodMatches;
    if (matcher.empty()) return goodMatches;
    
    // TODO: Implement FLANN matching with Ratio Test
    // 1. matcher->knnMatch(query, train, knnMatches, 2)
    // 2. Filter using ratio test
    
    return goodMatches;
}

} // namespace cv_curriculum
