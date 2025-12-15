#include "flann_matcher.hpp"

namespace cv_curriculum {

cv::Ptr<cv::DescriptorMatcher> createFlannLshMatcher() {
    // LSH params: table_number (12), key_size (20), multi_probe_level (2)
    // These are standard recommended values for ORB
    auto indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
    auto searchParams = cv::makePtr<cv::flann::SearchParams>(50); // Checks 50 trees/buckets
    
    return cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
}

std::vector<cv::DMatch> matchFlann(
    cv::Ptr<cv::DescriptorMatcher> matcher,
    const cv::Mat& query,
    const cv::Mat& train,
    float ratio)
{
    std::vector<cv::DMatch> goodMatches;
    if (query.empty() || train.empty()) return goodMatches;
    
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(query, train, knnMatches, 2);
    
    for (const auto& matches : knnMatches) {
        if (matches.size() >= 2) {
            if (matches[0].distance < ratio * matches[1].distance) {
                goodMatches.push_back(matches[0]);
            }
        }
    }
    
    return goodMatches;
}

} // namespace cv_curriculum
