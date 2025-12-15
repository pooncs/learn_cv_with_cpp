#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

namespace cv_curriculum {

/**
 * @brief Inverts image colors sequentially.
 */
void invertImageSequential(cv::Mat& img);

/**
 * @brief Inverts image colors in parallel using std::thread.
 * @param img Image to process.
 * @param numThreads Number of threads to use.
 */
void invertImageParallel(cv::Mat& img, int numThreads);

} // namespace cv_curriculum
