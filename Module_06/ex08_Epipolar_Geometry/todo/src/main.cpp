#include <iostream>
#include <opencv2/opencv.hpp>
#include "epipolar.hpp"

int main(int argc, char** argv) {
    std::cout << "Epipolar Geometry Exercise" << std::endl;
    
    std::vector<cv::Point2f> pts1, pts2;
    for(int i=0; i<20; ++i) {
        pts1.emplace_back(i*10, i*10);
        pts2.emplace_back(i*10 + 5, i*10); // Horizontal shift
    }
    
    auto result = cv_curriculum::computeFundamentalMatrix(pts1, pts2);
    
    if (result.F.empty()) {
        std::cout << "F not computed. Implement the function!" << std::endl;
    } else {
        std::cout << "F found." << std::endl;
    }
    
    return 0;
}
