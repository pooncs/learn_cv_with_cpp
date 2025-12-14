#include "harris.hpp"

namespace cv_curriculum {

cv::Mat computeHarrisResponse(const cv::Mat& gray, const HarrisConfig& config) {
    cv::Mat response = cv::Mat::zeros(gray.size(), CV_32F);
    
    // TODO: Implement Harris Corner Response
    // 1. Compute Gradients (Ix, Iy) using cv::Sobel
    // 2. Compute Products (Ixx, Iyy, Ixy)
    // 3. Compute Structure Tensor components (Gaussian Blur)
    // 4. Compute R = det(M) - k * trace(M)^2
    
    return response;
}

std::vector<cv::Point2f> detectCorners(const cv::Mat& response, float threshold) {
    std::vector<cv::Point2f> corners;
    
    // TODO: Implement Non-Maximum Suppression
    // 1. Iterate over pixels (avoiding borders)
    // 2. If value > threshold, check if it is a local maximum in 3x3 window
    // 3. If yes, add to corners
    
    return corners;
}

} // namespace cv_curriculum
