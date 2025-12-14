#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<int> compute_histogram(const cv::Mat& src);
cv::Mat draw_histogram(const std::vector<int>& hist, int width=512, int height=400);
cv::Mat equalize_hist_manual(const cv::Mat& src);
