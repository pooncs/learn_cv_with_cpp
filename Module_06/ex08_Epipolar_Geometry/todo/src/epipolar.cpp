#include "epipolar.hpp"

namespace cv_curriculum {

EpipolarResult computeFundamentalMatrix(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2)
{
    EpipolarResult result;
    
    // TODO: Implement Fundamental Matrix Calculation
    // 1. Check if enough points (>7 or 8)
    // 2. cv::findFundamentalMat with FM_RANSAC
    // 3. Filter inliers using the returned status mask
    
    return result;
}

void drawEpipolarLines(
    cv::Mat& img, 
    const std::vector<cv::Vec3f>& lines, 
    const std::vector<cv::Point2f>& pts)
{
    // TODO: Draw lines
    // Line equation: ax + by + c = 0
    // Calculate intersection with left border (x=0) and right border (x=cols)
    // cv::line(...)
}

} // namespace cv_curriculum
