#include "ratio_test.hpp"
#include <limits>
#include <algorithm>

namespace cv_curriculum {

static int countSetBits(uchar n) {
    int count = 0;
    while (n > 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}

static int computeHammingDistance(const cv::Mat& a, const cv::Mat& b) {
    if (a.cols != b.cols || a.type() != b.type()) return -1;
    int dist = 0;
    const uchar* ptrA = a.ptr<uchar>(0);
    const uchar* ptrB = b.ptr<uchar>(0);
    for (int i = 0; i < a.cols; ++i) {
        dist += countSetBits(ptrA[i] ^ ptrB[i]);
    }
    return dist;
}

std::vector<std::vector<cv::DMatch>> matchKnnBruteForce(
    const cv::Mat& queryDescriptors, 
    const cv::Mat& trainDescriptors, 
    int k) 
{
    std::vector<std::vector<cv::DMatch>> knnMatches;
    if (queryDescriptors.empty() || trainDescriptors.empty() || trainDescriptors.rows < k) {
        return knnMatches;
    }
    
    knnMatches.resize(queryDescriptors.rows);
    
    for (int i = 0; i < queryDescriptors.rows; ++i) {
        cv::Mat query = queryDescriptors.row(i);
        
        // Compute all distances
        std::vector<cv::DMatch> allMatches;
        allMatches.reserve(trainDescriptors.rows);
        
        for (int j = 0; j < trainDescriptors.rows; ++j) {
            int dist = computeHammingDistance(query, trainDescriptors.row(j));
            allMatches.emplace_back(i, j, static_cast<float>(dist));
        }
        
        // Sort
        std::partial_sort(allMatches.begin(), allMatches.begin() + k, allMatches.end());
        
        // Take top k
        for (int m = 0; m < k; ++m) {
            knnMatches[i].push_back(allMatches[m]);
        }
    }
    
    return knnMatches;
}

std::vector<cv::DMatch> filterRatioTest(
    const std::vector<std::vector<cv::DMatch>>& knnMatches, 
    float ratio) 
{
    std::vector<cv::DMatch> goodMatches;
    for (const auto& matches : knnMatches) {
        if (matches.size() >= 2) {
            float dist1 = matches[0].distance;
            float dist2 = matches[1].distance;
            
            if (dist1 < ratio * dist2) {
                goodMatches.push_back(matches[0]);
            }
        }
    }
    return goodMatches;
}

} // namespace cv_curriculum
