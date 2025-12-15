#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_curriculum {

/**
 * @brief Stitches two images into a panorama.
 * @param img1 Left image (reference).
 * @param img2 Right image (to be warped).
 * @return Stitched panorama.
 */
cv::Mat stitchImages(const cv::Mat& img1, const cv::Mat& img2);

} // namespace cv_curriculum
