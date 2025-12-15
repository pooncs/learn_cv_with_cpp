#include <iostream>
#include <opencv2/opencv.hpp>
#include "parallel_process.hpp"

int main(int argc, char** argv) {
    std::cout << "std::thread Exercise" << std::endl;
    
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::randn(img, 128, 50);
    cv::Mat imgRef = img.clone();
    
    cv_curriculum::invertImageSequential(imgRef);
    cv_curriculum::invertImageParallel(img, 2);
    
    double diff = cv::norm(img, imgRef, cv::NORM_L1);
    if (diff == 0) std::cout << "Success!" << std::endl;
    else std::cout << "Failure. Images differ." << std::endl;
    
    return 0;
}
