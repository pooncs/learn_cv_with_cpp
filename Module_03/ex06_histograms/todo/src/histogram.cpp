#include "histogram.hpp"

std::vector<int> compute_histogram(const cv::Mat& src) {
    // TODO: Compute histogram (256 bins)
    return std::vector<int>(256, 0);
}

cv::Mat draw_histogram(const std::vector<int>& hist, int width, int height) {
    // TODO: Draw histogram on a black image
    return cv::Mat::zeros(height, width, CV_8UC1);
}

cv::Mat equalize_hist_manual(const cv::Mat& src) {
    // TODO:
    // 1. Compute Hist
    // 2. Compute CDF
    // 3. Normalize CDF to [0, 255]
    // 4. Map pixels
    return src.clone();
}
