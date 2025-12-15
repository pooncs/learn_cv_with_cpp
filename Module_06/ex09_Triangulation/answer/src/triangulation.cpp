#include "triangulation.hpp"

namespace cv_curriculum {

std::vector<cv::Point3f> triangulateStereo(
    const cv::Mat& P1,
    const cv::Mat& P2,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2)
{
    std::vector<cv::Point3f> points3D;
    if (pts1.empty() || pts1.size() != pts2.size()) return points3D;
    
    cv::Mat pts1Mat(2, pts1.size(), CV_32F);
    cv::Mat pts2Mat(2, pts2.size(), CV_32F);
    
    for (size_t i = 0; i < pts1.size(); ++i) {
        pts1Mat.at<float>(0, i) = pts1[i].x;
        pts1Mat.at<float>(1, i) = pts1[i].y;
        pts2Mat.at<float>(0, i) = pts2[i].x;
        pts2Mat.at<float>(1, i) = pts2[i].y;
    }
    
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1Mat, pts2Mat, points4D);
    
    // Convert to 3D
    for (int i = 0; i < points4D.cols; ++i) {
        float w = points4D.at<float>(3, i);
        if (std::abs(w) > 1e-6) {
            float x = points4D.at<float>(0, i) / w;
            float y = points4D.at<float>(1, i) / w;
            float z = points4D.at<float>(2, i) / w;
            points3D.emplace_back(x, y, z);
        } else {
            // Point at infinity
            points3D.emplace_back(0, 0, 0); // Handle appropriately
        }
    }
    
    return points3D;
}

} // namespace cv_curriculum
