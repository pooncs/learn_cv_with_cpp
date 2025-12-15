#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

/**
 * @brief Creates a FLANN matcher suitable for ORB (LSH Index).
 * @return Ptr to DescriptorMatcher.
 */
cv::Ptr<cv::DescriptorMatcher> createFlannLshMatcher();

/**
 * @brief Matches using FLANN and filters with Ratio Test.
 * @param matcher The FLANN matcher.
 * @param query Descriptors.
 * @param train Descriptors.
 * @param ratio Ratio test threshold.
 * @return Good matches.
 */
std::vector<cv::DMatch> matchFlann(
    cv::Ptr<cv::DescriptorMatcher> matcher,
    const cv::Mat& query,
    const cv::Mat& train,
    float ratio = 0.75f);

} // namespace cv_curriculum
