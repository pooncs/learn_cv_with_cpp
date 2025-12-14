#include <iostream>
#include "color_spaces.hpp"

int main() {
    // Create simple image: Left half Blue, Right half Red
    cv::Mat img(100, 200, CV_8UC3);
    img(cv::Rect(0, 0, 100, 100)) = cv::Scalar(255, 0, 0); // Blue
    img(cv::Rect(100, 0, 100, 100)) = cv::Scalar(0, 0, 255); // Red

    cv::Mat gray = to_gray(img);
    cv::Mat hsv = to_hsv(img);

    // Check center of left half (Blue)
    if (!hsv.empty() && !gray.empty()) {
        cv::Vec3b hsv_blue = hsv.at<cv::Vec3b>(50, 50);
        std::cout << "Blue Pixel -> Gray: " << (int)gray.at<uchar>(50, 50) << "\n";
        std::cout << "Blue Pixel -> HSV: " << (int)hsv_blue[0] << "," << (int)hsv_blue[1] << "," << (int)hsv_blue[2] << "\n";
    } else {
        std::cout << "Not implemented yet.\n";
    }

    return 0;
}
