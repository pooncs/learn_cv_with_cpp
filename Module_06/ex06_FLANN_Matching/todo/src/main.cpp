#include <iostream>
#include <opencv2/opencv.hpp>
#include "flann_matcher.hpp"

int main(int argc, char** argv) {
    std::cout << "FLANN Matching Exercise" << std::endl;
    
    cv::Mat query = cv::Mat::zeros(10, 32, CV_8UC1);
    cv::randn(query, 128, 50);
    cv::Mat train = cv::Mat::zeros(10, 32, CV_8UC1);
    cv::randn(train, 128, 50);
    
    auto matcher = cv_curriculum::createFlannLshMatcher();
    if (!matcher) {
        std::cout << "Matcher creation not implemented." << std::endl;
        return 0;
    }
    
    // Check if it's actually Flann
    // Note: dynamic_cast might fail if RTTI is off, but OpenCV usually has it.
    
    auto matches = cv_curriculum::matchFlann(matcher, query, train);
    std::cout << "Matches found: " << matches.size() << std::endl;
    
    return 0;
}
