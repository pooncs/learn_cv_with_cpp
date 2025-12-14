#include <iostream>
#include "morphology.hpp"

int main() {
    // 5x5 image with a 3x3 block in center
    cv::Mat img = cv::Mat::zeros(5, 5, CV_8UC1);
    // Set 3x3 center to 255
    for(int i=1; i<=3; ++i)
        for(int j=1; j<=3; ++j)
            img.at<uchar>(i, j) = 255;

    // Add a hole at 2,2
    img.at<uchar>(2, 2) = 0;

    std::cout << "Input:\n" << img << "\n\n";

    cv::Mat dilated = my_dilate(img);
    std::cout << "Dilated (Hole should fill):\n" << dilated << "\n\n";

    cv::Mat eroded = my_erode(img);
    std::cout << "Eroded (Block should shrink):\n" << eroded << "\n\n";

    return 0;
}
