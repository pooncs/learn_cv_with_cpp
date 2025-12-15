#include "matcher.hpp"

namespace cv_curriculum {

int computeHammingDistance(const cv::Mat& a, const cv::Mat& b) {
    // TODO: Implement Hamming Distance
    // 1. Iterate over bytes in the row
    // 2. XOR the bytes
    // 3. Count set bits (population count)
    // 4. Accumulate
    return -1;
}

std::vector<cv::DMatch> matchBruteForce(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors) {
    std::vector<cv::DMatch> matches;
    
    // TODO: Implement Brute Force Matching
    // 1. Iterate over each query descriptor
    // 2. For each query, iterate over all train descriptors
    // 3. Find the one with minimum Hamming distance
    // 4. Create DMatch object and add to list
    
    return matches;
}

} // namespace cv_curriculum
