#include "triangulation.hpp"

namespace cv_curriculum {

std::vector<cv::Point3f> triangulateStereo(
    const cv::Mat& P1,
    const cv::Mat& P2,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2)
{
    std::vector<cv::Point3f> points3D;
    
    // TODO: Implement Triangulation
    // 1. Convert points to appropriate format (2xN Mat)
    // 2. cv::triangulatePoints(P1, P2, pts1, pts2, points4D)
    // 3. Convert Homogeneous (4D) to Euclidean (3D) by dividing by w.
    
    return points3D;
}

} // namespace cv_curriculum
