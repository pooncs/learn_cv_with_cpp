#include "ratio_test.hpp"

namespace cv_curriculum {

std::vector<std::vector<cv::DMatch>> matchKnnBruteForce(
    const cv::Mat& queryDescriptors, 
    const cv::Mat& trainDescriptors, 
    int k) 
{
    std::vector<std::vector<cv::DMatch>> knnMatches;
    
    // TODO: Implement KNN Brute Force
    // 1. For each query descriptor:
    //    a. Calculate distance to ALL train descriptors.
    //    b. Sort distances.
    //    c. Keep top k.
    
    return knnMatches;
}

std::vector<cv::DMatch> filterRatioTest(
    const std::vector<std::vector<cv::DMatch>>& knnMatches, 
    float ratio) 
{
    std::vector<cv::DMatch> goodMatches;
    
    // TODO: Implement Lowe's Ratio Test
    // 1. Iterate over knnMatches.
    // 2. Ensure at least 2 neighbors exist.
    // 3. Check if best_dist < ratio * second_best_dist.
    // 4. If yes, add best match to result.
    
    return goodMatches;
}

} // namespace cv_curriculum
