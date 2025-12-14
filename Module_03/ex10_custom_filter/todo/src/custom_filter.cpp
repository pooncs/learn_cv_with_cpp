#include "custom_filter.hpp"

cv::Mat get_gaussian_kernel(int ksize, double sigma) {
    // TODO:
    // 1. Alloc 1xN float matrix
    // 2. Fill with Gaussian values
    // 3. Normalize sum to 1.0
    return cv::Mat();
}

cv::Mat apply_separable_filter(const cv::Mat& src, const cv::Mat& kernelX, const cv::Mat& kernelY) {
    // TODO:
    // 1. Filter rows (src * kernelX) -> temp
    // 2. Filter cols (temp * kernelY) -> dst
    // Hint: use cv::filter2D
    return cv::Mat();
}
