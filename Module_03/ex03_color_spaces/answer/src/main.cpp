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
    // Blue: B=255, G=0, R=0 -> Gray = 0.114*255 = 29
    // HSV: H=240/2=120, S=255, V=255
    cv::Vec3b hsv_blue = hsv.at<cv::Vec3b>(50, 50);
    std::cout << "Blue Pixel -> Gray: " << (int)gray.at<uchar>(50, 50) << "\n";
    std::cout << "Blue Pixel -> HSV: " << (int)hsv_blue[0] << "," << (int)hsv_blue[1] << "," << (int)hsv_blue[2] << "\n";

    return 0;
}
