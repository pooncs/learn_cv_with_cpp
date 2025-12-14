#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Computes H such that dst = H * src
cv::Mat compute_homography(const std::vector<cv::Point2f>& src_pts, const std::vector<cv::Point2f>& dst_pts);

// Warps src_img to dst_img using H
cv::Mat warp_image(const cv::Mat& src_img, const cv::Mat& H, const cv::Size& dsize);
