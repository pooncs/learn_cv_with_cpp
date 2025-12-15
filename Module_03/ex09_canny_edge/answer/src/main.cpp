#include <iostream>
#include "canny_utils.hpp"

int main() {
    // Create an image with a thick edge
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC1);
    // Ramp edge: 0, 50, 100, 150, 200, 255...
    for(int j=40; j<60; ++j) {
        img.col(j) = (j-40) * 12;
    }
    img.colRange(60, 100).setTo(255);

    // Compute Gradients
    cv::Mat gray;
    img.convertTo(gray, CV_32F);
    cv::Mat Gx, Gy;
    cv::Sobel(gray, Gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, Gy, CV_32F, 0, 1, 3);
    
    cv::Mat mag, angle;
    cv::cartToPolar(Gx, Gy, mag, angle, true);

    cv::Mat nms = non_max_suppression(mag, angle);

    std::cout << "Computed NMS.\n";
    // cv::imshow("Mag", mag / 255.0); // Visualize mag
    // cv::imshow("NMS", nms);
    // cv::waitKey(0);

    return 0;
}
