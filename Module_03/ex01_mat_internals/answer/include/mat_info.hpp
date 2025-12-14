#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// Returns a formatted string with Mat properties
std::string get_mat_info(const cv::Mat& m);

// Checks if two Mats share the same data pointer
bool share_data(const cv::Mat& m1, const cv::Mat& m2);
