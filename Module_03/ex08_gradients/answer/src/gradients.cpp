#include "gradients.hpp"
#include <cmath>

std::pair<cv::Mat, cv::Mat> compute_sobel(const cv::Mat& src) {
    cv::Mat gray;
    if(src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src;

    cv::Mat Gx, Gy;
    // Use CV_32F to avoid overflow/underflow
    cv::Sobel(gray, Gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, Gy, CV_32F, 0, 1, 3);

    return {Gx, Gy};
}

std::pair<cv::Mat, cv::Mat> compute_magnitude_angle(const cv::Mat& Gx, const cv::Mat& Gy) {
    cv::Mat mag, angle;
    // cartToPolar handles magnitude and angle calculation efficiently
    // angleInDegrees = true
    cv::cartToPolar(Gx, Gy, mag, angle, true);
    return {mag, angle};
}
