#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
    std::cout << "Image size: " << img.size() << std::endl;
    return 0;
}
