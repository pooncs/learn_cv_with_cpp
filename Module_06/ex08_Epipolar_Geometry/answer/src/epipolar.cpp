#include "epipolar.hpp"
#include <random>

namespace cv_curriculum {

EpipolarResult computeFundamentalMatrix(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2)
{
    EpipolarResult result;
    if (pts1.size() < 8 || pts2.size() < 8) return result;
    
    std::vector<uchar> status;
    // RANSAC threshold 1.0 to 3.0 usually
    result.F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, status);
    
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            result.inliers1.push_back(pts1[i]);
            result.inliers2.push_back(pts2[i]);
        }
    }
    
    return result;
}

void drawEpipolarLines(
    cv::Mat& img, 
    const std::vector<cv::Vec3f>& lines, 
    const std::vector<cv::Point2f>& pts)
{
    if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::Scalar color(dist(rng), dist(rng), dist(rng));
        cv::Vec3f l = lines[i];
        // Line: ax + by + c = 0
        // x0 = 0 -> y0 = -c/b
        // x1 = w -> y1 = -(c + a*w)/b
        
        double a = l[0], b = l[1], c = l[2];
        
        cv::Point pt1, pt2;
        pt1.x = 0;
        pt1.y = cvRound(-c / b);
        pt2.x = img.cols;
        pt2.y = cvRound(-(c + a * img.cols) / b);
        
        cv::line(img, pt1, pt2, color, 1);
        if (i < pts.size()) {
            cv::circle(img, pts[i], 5, color, -1);
        }
    }
}

} // namespace cv_curriculum
