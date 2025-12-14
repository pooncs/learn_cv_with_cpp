#include "custom_filter.hpp"
#include <cmath>
#include <numeric>

cv::Mat get_gaussian_kernel(int ksize, double sigma) {
    CV_Assert(ksize % 2 == 1);
    cv::Mat kernel(1, ksize, CV_32F);
    float* ptr = kernel.ptr<float>(0);
    int center = ksize / 2;
    float sum = 0.0f;

    for(int i=0; i<ksize; ++i) {
        int x = i - center;
        float val = std::exp(-(x*x) / (2 * sigma * sigma));
        ptr[i] = val;
        sum += val;
    }

    // Normalize
    kernel /= sum;
    return kernel;
}

cv::Mat apply_separable_filter(const cv::Mat& src, const cv::Mat& kernelX, const cv::Mat& kernelY) {
    // We can use cv::sepFilter2D which is highly optimized, 
    // or implement manually. Let's implement manually for learning.
    
    // 1. Convolve Rows (Horizontal)
    // Output should be float to prevent overflow/rounding before next step
    cv::Mat intermediate; 
    // Use filter2D for 1D convolution
    // anchor default (-1,-1) means center
    cv::filter2D(src, intermediate, CV_32F, kernelX);

    // 2. Convolve Cols (Vertical)
    cv::Mat dst;
    // kernelY is 1xN, filter2D expects kernel. 
    // If we want vertical, we transpose it or pass it as is?
    // filter2D scans kernel over image.
    // To filter vertical, we need a Nx1 kernel.
    cv::Mat kY_t;
    if (kernelY.rows == 1) kY_t = kernelY.t();
    else kY_t = kernelY;

    cv::filter2D(intermediate, dst, src.type(), kY_t);

    return dst;
}
