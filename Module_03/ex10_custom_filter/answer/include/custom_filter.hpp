#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Generates a 1D Gaussian kernel of size ksize (odd)
// Returns 1xN matrix (CV_32F)
cv::Mat get_gaussian_kernel(int ksize, double sigma);

// Applies separable filtering
// kernelX: 1xN
// kernelY: 1xM (usually Nx1 or 1xN depending on implementation, let's assume 1D vectors)
cv::Mat apply_separable_filter(const cv::Mat& src, const cv::Mat& kernelX, const cv::Mat& kernelY);
