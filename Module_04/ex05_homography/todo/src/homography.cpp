#include "homography.hpp"

cv::Mat compute_homography(const std::vector<cv::Point2f>& src_pts, const std::vector<cv::Point2f>& dst_pts) {
    // TODO: Compute Homography using cv::findHomography
    return cv::Mat();
}

cv::Mat warp_image(const cv::Mat& src_img, const cv::Mat& H, const cv::Size& dsize) {
    // TODO: Warp image using cv::warpPerspective
    return cv::Mat();
}
