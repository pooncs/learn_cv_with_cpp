#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "parallel_process.hpp"

int main(int argc, char** argv) {
    // Create large image
    int width = 4096;
    int height = 4096;
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
    cv::randn(img, 128, 50);
    
    cv::Mat imgSeq = img.clone();
    cv::Mat imgPar = img.clone();
    
    // Sequential
    auto start = std::chrono::high_resolution_clock::now();
    cv_curriculum::invertImageSequential(imgSeq);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dtSeq = end - start;
    std::cout << "Sequential: " << dtSeq.count() << " ms" << std::endl;
    
    // Parallel
    int threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 4;
    std::cout << "Using " << threads << " threads." << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    cv_curriculum::invertImageParallel(imgPar, threads);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dtPar = end - start;
    std::cout << "Parallel:   " << dtPar.count() << " ms" << std::endl;
    
    std::cout << "Speedup: " << dtSeq.count() / dtPar.count() << "x" << std::endl;
    
    // Check equality
    double diff = cv::norm(imgSeq, imgPar, cv::NORM_L1);
    if (diff == 0) std::cout << "Results match!" << std::endl;
    else std::cout << "Results differ! Error." << std::endl;
    
    return 0;
}
