#include "pnp.hpp"

std::pair<cv::Mat, cv::Mat> estimate_pose(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& K,
    const cv::Mat& dist_coeffs) 
{
    cv::Mat rvec, tvec;
    cv::solvePnP(object_points, image_points, K, dist_coeffs, rvec, tvec);
    return {rvec, tvec};
}

void draw_axes(cv::Mat& img, const cv::Mat& K, const cv::Mat& dist_coeffs, const cv::Mat& rvec, const cv::Mat& tvec, float length) {
    std::vector<cv::Point3f> axis = {
        {0, 0, 0}, 
        {length, 0, 0}, 
        {0, length, 0}, 
        {0, 0, -length} // Z is forward, but usually we draw up as -Y or Z depending on definition. Let's draw standard XYZ
    };
    // OpenCV Camera: X Right, Y Down, Z Forward.
    // If object is on Z=0 plane (like chessboard), then Z-up in object frame is -Z in camera?
    // Let's assume standard object frame.
    
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(axis, rvec, tvec, K, dist_coeffs, img_pts);

    cv::line(img, img_pts[0], img_pts[1], cv::Scalar(0, 0, 255), 2); // X - Red
    cv::line(img, img_pts[0], img_pts[2], cv::Scalar(0, 255, 0), 2); // Y - Green
    cv::line(img, img_pts[0], img_pts[3], cv::Scalar(255, 0, 0), 2); // Z - Blue
}
