#include <iostream>
#include <opencv2/opencv.hpp>
#include "ratio_test.hpp"

int main(int argc, char** argv) {
    std::cout << "Ratio Test Exercise" << std::endl;
    
    // Create dummy descriptors
    // Q0 matches T0 (dist 0) and T1 (dist 2) -> Ratio 0/2 = 0. Good.
    // Q1 matches T2 (dist 5) and T3 (dist 5) -> Ratio 5/5 = 1. Bad.
    
    cv::Mat query = cv::Mat::zeros(2, 32, CV_8UC1);
    cv::Mat train = cv::Mat::zeros(4, 32, CV_8UC1);
    
    // Q0: 0000...
    // T0: 0000... (dist 0)
    // T1: 0011... (dist 2)
    train.at<uchar>(1, 0) = 0x03; 
    
    // Q1: 1111...
    query.at<uchar>(1, 0) = 0x0F;
    // T2: 1111... (dist 0? No wait, let's make it dist 5)
    // T2: 1111... ^ 00000101 (dist 2)
    // Let's make Q1 have dist 10 to T2 and 11 to T3
    // Q1 = 00...
    // T2 = 000011111100 (10 bits)
    // T3 = 000011111110 (11 bits)
    // It's easier to just trust the KNN logic or mock it, but we need to run matchKnnBruteForce first.
    
    // Let's rely on unit tests for precise logic verification.
    // This main is just a stub runner.
    
    auto knn = cv_curriculum::matchKnnBruteForce(query, train, 2);
    if (knn.empty()) {
        std::cout << "KNN returned empty. Implement it!" << std::endl;
        return 0;
    }
    
    auto good = cv_curriculum::filterRatioTest(knn, 0.7f);
    std::cout << "Good matches: " << good.size() << std::endl;
    
    return 0;
}
