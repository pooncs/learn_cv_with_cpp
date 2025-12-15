#include <iostream>
#include <opencv2/opencv.hpp>
#include "matcher.hpp"

int main(int argc, char** argv) {
    std::cout << "Brute Force Matching Exercise" << std::endl;
    
    // Create dummy descriptors
    // Query: 11110000...
    // Train1: 11110000... (dist 0)
    // Train2: 00001111... (dist 8)
    
    cv::Mat query = cv::Mat::zeros(1, 32, CV_8UC1);
    query.at<uchar>(0, 0) = 0xF0;
    
    cv::Mat train = cv::Mat::zeros(2, 32, CV_8UC1);
    train.at<uchar>(0, 0) = 0xF0; // Match
    train.at<uchar>(1, 0) = 0x0F; // No Match
    
    int dist = cv_curriculum::computeHammingDistance(query.row(0), train.row(0));
    std::cout << "Distance (expect 0): " << dist << std::endl;
    
    auto matches = cv_curriculum::matchBruteForce(query, train);
    if (matches.empty()) {
        std::cout << "Matches empty. Implement the function!" << std::endl;
    } else {
        std::cout << "Matched query 0 to train " << matches[0].trainIdx << " with dist " << matches[0].distance << std::endl;
    }
    
    return 0;
}
