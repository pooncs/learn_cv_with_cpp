#include "matcher.hpp"
#include <limits>
#include <bitset>

namespace cv_curriculum {

// Helper to count bits in a byte
// Could use std::bitset<8> or __builtin_popcount
static int countSetBits(uchar n) {
    int count = 0;
    while (n > 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}

int computeHammingDistance(const cv::Mat& a, const cv::Mat& b) {
    // Check dimensions
    if (a.cols != b.cols || a.type() != b.type()) return -1;
    
    int dist = 0;
    const uchar* ptrA = a.ptr<uchar>(0);
    const uchar* ptrB = b.ptr<uchar>(0);
    
    for (int i = 0; i < a.cols; ++i) {
        uchar valA = ptrA[i];
        uchar valB = ptrB[i];
        // XOR gives bits that are different
        uchar xorVal = valA ^ valB;
        dist += countSetBits(xorVal);
    }
    return dist;
}

std::vector<cv::DMatch> matchBruteForce(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors) {
    std::vector<cv::DMatch> matches;
    
    if (queryDescriptors.empty() || trainDescriptors.empty()) return matches;
    
    // For each query descriptor
    for (int i = 0; i < queryDescriptors.rows; ++i) {
        cv::Mat query = queryDescriptors.row(i);
        
        int bestIdx = -1;
        int bestDist = std::numeric_limits<int>::max();
        
        // Search through all train descriptors
        for (int j = 0; j < trainDescriptors.rows; ++j) {
            cv::Mat train = trainDescriptors.row(j);
            
            int dist = computeHammingDistance(query, train);
            
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = j;
            }
        }
        
        if (bestIdx != -1) {
            // DMatch(queryIdx, trainIdx, distance)
            matches.emplace_back(i, bestIdx, static_cast<float>(bestDist));
        }
    }
    
    return matches;
}

} // namespace cv_curriculum
