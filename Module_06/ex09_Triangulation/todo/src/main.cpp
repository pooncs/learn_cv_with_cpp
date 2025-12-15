#include <iostream>
#include <opencv2/opencv.hpp>
#include "triangulation.hpp"

int main(int argc, char** argv) {
    std::cout << "Triangulation Exercise" << std::endl;
    
    // Stub
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_32F);
    cv::Mat P2 = cv::Mat::eye(3, 4, CV_32F);
    std::vector<cv::Point2f> pts1 = {{0,0}};
    std::vector<cv::Point2f> pts2 = {{0,0}};
    
    auto res = cv_curriculum::triangulateStereo(P1, P2, pts1, pts2);
    if (res.empty()) {
        std::cout << "Triangulation empty. Implement it!" << std::endl;
    }
    
    return 0;
}
