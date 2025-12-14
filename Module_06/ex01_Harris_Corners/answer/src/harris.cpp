#include "harris.hpp"

namespace cv_curriculum {

cv::Mat computeHarrisResponse(const cv::Mat& gray, const HarrisConfig& config) {
    cv::Mat Ix, Iy;
    
    // 1. Compute Gradients
    cv::Sobel(gray, Ix, CV_32F, 1, 0, config.apertureSize);
    cv::Sobel(gray, Iy, CV_32F, 0, 1, config.apertureSize);

    // 2. Compute Products
    cv::Mat Ixx = Ix.mul(Ix);
    cv::Mat Iyy = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // 3. Gaussian Blur (Window Sum)
    cv::Mat Sxx, Syy, Sxy;
    int ksize = config.blockSize * 2 + 1; // Ensure odd
    double sigma = config.blockSize * 0.5; // Heuristic
    
    cv::GaussianBlur(Ixx, Sxx, cv::Size(ksize, ksize), sigma);
    cv::GaussianBlur(Iyy, Syy, cv::Size(ksize, ksize), sigma);
    cv::GaussianBlur(Ixy, Sxy, cv::Size(ksize, ksize), sigma);

    // 4. Compute Response
    cv::Mat response = cv::Mat::zeros(gray.size(), CV_32F);
    
    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            float sxx = Sxx.at<float>(y, x);
            float syy = Syy.at<float>(y, x);
            float sxy = Sxy.at<float>(y, x);
            
            float det = sxx * syy - sxy * sxy;
            float trace = sxx + syy;
            
            response.at<float>(y, x) = det - config.k * trace * trace;
        }
    }
    
    return response;
}

std::vector<cv::Point2f> detectCorners(const cv::Mat& response, float threshold) {
    std::vector<cv::Point2f> corners;
    
    // 3x3 Non-Maximum Suppression
    for (int y = 1; y < response.rows - 1; ++y) {
        for (int x = 1; x < response.cols - 1; ++x) {
            float val = response.at<float>(y, x);
            
            if (val > threshold) {
                // Check local maximum
                bool isMax = true;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        if (response.at<float>(y + dy, x + dx) >= val) {
                            isMax = false;
                            break;
                        }
                    }
                    if (!isMax) break;
                }
                
                if (isMax) {
                    corners.emplace_back(static_cast<float>(x), static_cast<float>(y));
                }
            }
        }
    }
    return corners;
}

} // namespace cv_curriculum
