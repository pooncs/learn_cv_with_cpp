#include <iostream>
#include <opencv2/opencv.hpp>
#include "robust_matcher.hpp"

int main(int argc, char** argv) {
    std::cout << "RANSAC Homography Exercise" << std::endl;
    
    // Create dummy points for a perfect homography
    // Identity transform
    std::vector<cv::Point2f> pts1 = {{0,0}, {10,0}, {0,10}, {10,10}, {100, 100}};
    std::vector<cv::Point2f> pts2 = {{0,0}, {10,0}, {0,10}, {10,10}, {500, 500}}; // Last one is outlier
    
    std::vector<cv::DMatch> matches;
    for(int i=0; i<5; ++i) matches.emplace_back(i, i, 0.0f);
    
    auto result = cv_curriculum::computeRobustHomography(pts1, pts2, matches, 1.0);
    
    if (result.H.empty()) {
        std::cout << "Homography not computed. Implement the function!" << std::endl;
    } else {
        std::cout << "Inliers: " << result.inlierMatches.size() << " (Expect 4)" << std::endl;
    }
    
    return 0;
}
