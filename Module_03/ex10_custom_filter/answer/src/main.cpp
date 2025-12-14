#include <iostream>
#include "custom_filter.hpp"

int main() {
    // Noisy image
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::randu(img, 0, 255);

    // Create kernels
    int ksize = 7;
    double sigma = 1.5;
    cv::Mat kX = get_gaussian_kernel(ksize, sigma);
    cv::Mat kY = kX; // Symmetric

    // Apply
    cv::Mat res = apply_separable_filter(img, kX, kY);

    // Compare with OpenCV GaussianBlur
    cv::Mat ref;
    cv::GaussianBlur(img, ref, cv::Size(ksize, ksize), sigma);

    // Compute diff
    cv::Mat diff;
    cv::absdiff(res, ref, diff);
    double minVal, maxVal;
    cv::minMaxLoc(diff, &minVal, &maxVal);

    std::cout << "Max difference from cv::GaussianBlur: " << maxVal << "\n";
    // Should be very small (rounding errors)

    return 0;
}
