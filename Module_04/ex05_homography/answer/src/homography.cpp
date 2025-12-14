#include "homography.hpp"

cv::Mat compute_homography(const std::vector<cv::Point2f>& src_pts, const std::vector<cv::Point2f>& dst_pts) {
    CV_Assert(src_pts.size() >= 4 && dst_pts.size() >= 4);
    // Use RANSAC for robustness, though 0 is fine for perfect points
    return cv::findHomography(src_pts, dst_pts, cv::RANSAC);
}

cv::Mat warp_image(const cv::Mat& src_img, const cv::Mat& H, const cv::Size& dsize) {
    cv::Mat dst;
    cv::warpPerspective(src_img, dst, H, dsize);
    return dst;
}
