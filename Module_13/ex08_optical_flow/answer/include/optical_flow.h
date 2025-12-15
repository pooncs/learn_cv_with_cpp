#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace cv_tracking {

/**
 * @brief Optical Flow Tracker using Lucas-Kanade method.
 * 
 * This class provides methods to compute sparse optical flow between two frames.
 * It includes a custom implementation of the single-level Lucas-Kanade algorithm
 * for educational purposes, and a wrapper around OpenCV's pyramidal implementation
 * for robust tracking.
 */
class OpticalFlowTracker {
public:
    OpticalFlowTracker();
    ~OpticalFlowTracker();

    /**
     * @brief Compute optical flow for specific points using custom single-level LK implementation.
     * 
     * @param prevImg Previous frame (grayscale)
     * @param nextImg Current frame (grayscale)
     * @param prevPts Points in previous frame to track
     * @param nextPts Output tracked points in current frame
     * @param status Output status for each point (1 if tracked, 0 otherwise)
     * @param winSize Window size for integration (default 21x21)
     */
    void computeFlowCustom(const cv::Mat& prevImg, const cv::Mat& nextImg,
                           const std::vector<cv::Point2f>& prevPts,
                           std::vector<cv::Point2f>& nextPts,
                           std::vector<uchar>& status,
                           cv::Size winSize = cv::Size(21, 21));

    /**
     * @brief Compute optical flow using OpenCV's Pyramidal LK implementation.
     * 
     * @param prevImg Previous frame
     * @param nextImg Current frame
     * @param prevPts Points in previous frame
     * @param nextPts Output tracked points
     * @param status Output status
     */
    void computeFlowOpenCV(const cv::Mat& prevImg, const cv::Mat& nextImg,
                           const std::vector<cv::Point2f>& prevPts,
                           std::vector<cv::Point2f>& nextPts,
                           std::vector<uchar>& status);

private:
    // Helper to check if a point is within image bounds
    bool isInside(const cv::Mat& img, const cv::Point2f& pt, const cv::Size& winSize);
    
    // Get subpixel value using bilinear interpolation
    float getPixelValue(const cv::Mat& img, float x, float y);
};

} // namespace cv_tracking
