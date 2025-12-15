#include "optical_flow.h"
#include <iostream>

namespace cv_tracking {

OpticalFlowTracker::OpticalFlowTracker() {}

OpticalFlowTracker::~OpticalFlowTracker() {}

bool OpticalFlowTracker::isInside(const cv::Mat& img, const cv::Point2f& pt, const cv::Size& winSize) {
    int halfWinW = winSize.width / 2;
    int halfWinH = winSize.height / 2;
    return (pt.x - halfWinW >= 0 && pt.x + halfWinW < img.cols &&
            pt.y - halfWinH >= 0 && pt.y + halfWinH < img.rows);
}

float OpticalFlowTracker::getPixelValue(const cv::Mat& img, float x, float y) {
    if (x < 0 || x >= img.cols - 1 || y < 0 || y >= img.rows - 1)
        return 0.0f;

    int x0 = (int)x;
    int y0 = (int)y;
    float dx = x - x0;
    float dy = y - y0;

    float v00 = (float)img.at<uchar>(y0, x0);
    float v01 = (float)img.at<uchar>(y0, x0 + 1);
    float v10 = (float)img.at<uchar>(y0 + 1, x0);
    float v11 = (float)img.at<uchar>(y0 + 1, x0 + 1);

    return (1.0f - dx) * (1.0f - dy) * v00 +
           dx * (1.0f - dy) * v01 +
           (1.0f - dx) * dy * v10 +
           dx * dy * v11;
}

void OpticalFlowTracker::computeFlowCustom(const cv::Mat& prevImg, const cv::Mat& nextImg,
                                           const std::vector<cv::Point2f>& prevPts,
                                           std::vector<cv::Point2f>& nextPts,
                                           std::vector<uchar>& status,
                                           cv::Size winSize) {
    nextPts.resize(prevPts.size());
    status.resize(prevPts.size());

    // Pre-compute gradients of the previous image
    cv::Mat Ix, Iy;
    cv::Sobel(prevImg, Ix, CV_32F, 1, 0, 3);
    cv::Sobel(prevImg, Iy, CV_32F, 0, 1, 3);

    int halfWinW = winSize.width / 2;
    int halfWinH = winSize.height / 2;

    for (size_t i = 0; i < prevPts.size(); ++i) {
        cv::Point2f pt = prevPts[i];
        
        // Check if point is too close to boundary
        if (!isInside(prevImg, pt, winSize)) {
            status[i] = 0;
            continue;
        }

        // Initialize displacement guess (0,0) or use previous frame's velocity if available
        // Here we assume small motion, start at prev position
        cv::Point2f currPt = pt;
        
        // Iterative Lucas-Kanade
        const int MAX_ITER = 20;
        const float EPSILON = 0.01f;
        bool converged = false;

        // Pre-compute G matrix (spatial gradient matrix)
        // G = sum([Ix^2  IxIy]
        //         [IxIy  Iy^2])
        float Gxx = 0, Gyy = 0, Gxy = 0;
        for (int wy = -halfWinH; wy <= halfWinH; ++wy) {
            for (int wx = -halfWinW; wx <= halfWinW; ++wx) {
                float ix = Ix.at<float>(pt.y + wy, pt.x + wx);
                float iy = Iy.at<float>(pt.y + wy, pt.x + wx);
                Gxx += ix * ix;
                Gyy += iy * iy;
                Gxy += ix * iy;
            }
        }

        // Determinant of G
        float det = Gxx * Gyy - Gxy * Gxy;
        if (std::abs(det) < 1e-6) {
            status[i] = 0; // Ill-conditioned (aperture problem)
            continue;
        }
        
        // Invert G
        float invDet = 1.0f / det;
        float Ginv_xx = Gyy * invDet;
        float Ginv_yy = Gxx * invDet;
        float Ginv_xy = -Gxy * invDet;

        for (int iter = 0; iter < MAX_ITER; ++iter) {
            if (!isInside(nextImg, currPt, winSize)) {
                break;
            }

            // Compute image difference vector b
            // b = sum([Ix * It]
            //         [Iy * It])
            // where It = I(x) - J(x+u)   (difference between template and warped current)
            // Wait, usually It = J(x+u) - I(x).
            // Let's stick to: Minimize sum((J(x+u) - I(x))^2)
            // Taylor: J(x+u+du) approx J(x+u) + gradJ * du
            // J(x+u) + gradJ*du - I(x) = 0
            // gradJ * du = I(x) - J(x+u)
            // Here we approximate gradJ with gradI (from prev image) for efficiency (KLT).
            // So: [Ix Iy] * [du dv]^T = I(x) - J(x+u)
            
            float bx = 0, by = 0;
            
            for (int wy = -halfWinH; wy <= halfWinH; ++wy) {
                for (int wx = -halfWinW; wx <= halfWinW; ++wx) {
                    float ix = Ix.at<float>(pt.y + wy, pt.x + wx);
                    float iy = Iy.at<float>(pt.y + wy, pt.x + wx);
                    
                    float valPrev = (float)prevImg.at<uchar>(pt.y + wy, pt.x + wx);
                    float valNext = getPixelValue(nextImg, currPt.x + wx, currPt.y + wy);
                    
                    float diff = valPrev - valNext; // I(x) - J(x+u)
                    
                    bx += ix * diff;
                    by += iy * diff;
                }
            }

            float du = Ginv_xx * bx + Ginv_xy * by;
            float dv = Ginv_xy * bx + Ginv_yy * by;

            currPt.x += du;
            currPt.y += dv;

            if (du * du + dv * dv < EPSILON) {
                converged = true;
                break;
            }
        }

        nextPts[i] = currPt;
        status[i] = converged ? 1 : 0;
    }
}

void OpticalFlowTracker::computeFlowOpenCV(const cv::Mat& prevImg, const cv::Mat& nextImg,
                                           const std::vector<cv::Point2f>& prevPts,
                                           std::vector<cv::Point2f>& nextPts,
                                           std::vector<uchar>& status) {
    if (prevPts.empty()) return;
    
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err);
}

} // namespace cv_tracking
