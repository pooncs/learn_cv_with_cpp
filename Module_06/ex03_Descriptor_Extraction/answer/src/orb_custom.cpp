#include "orb_custom.hpp"
#include <random>
#include <cmath>

namespace cv_curriculum {

// Static pattern for BRIEF (256 pairs of points)
// Each point is (x, y) relative to center
static std::vector<cv::Point> patternA, patternB;
static bool patternInit = false;

static void initPattern(int patchSize) {
    if (patternInit) return;
    std::mt19937 rng(42); // Fixed seed
    // Gaussian distribution around center
    // Sigma usually patchSize / 5
    float sigma = patchSize / 5.0f;
    std::normal_distribution<float> dist(0, sigma);

    patternA.resize(256);
    patternB.resize(256);

    for (int i = 0; i < 256; ++i) {
        patternA[i] = cv::Point(std::round(dist(rng)), std::round(dist(rng)));
        patternB[i] = cv::Point(std::round(dist(rng)), std::round(dist(rng)));
        
        // Clamp to patch
        int limit = patchSize / 2;
        patternA[i].x = std::max(-limit, std::min(limit, patternA[i].x));
        patternA[i].y = std::max(-limit, std::min(limit, patternA[i].y));
        patternB[i].x = std::max(-limit, std::min(limit, patternB[i].x));
        patternB[i].y = std::max(-limit, std::min(limit, patternB[i].y));
    }
    patternInit = true;
}

void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, int patchSize) {
    int halfPatch = patchSize / 2;
    
    for (auto& kp : keypoints) {
        float m01 = 0, m10 = 0;
        int cx = std::round(kp.pt.x);
        int cy = std::round(kp.pt.y);
        
        // Circular mask usually used, here we iterate box for simplicity
        // But weight by distance? ORB uses simple moments in a circle.
        // Let's iterate -halfPatch to halfPatch
        for (int y = -halfPatch; y <= halfPatch; ++y) {
            for (int x = -halfPatch; x <= halfPatch; ++x) {
                if (x*x + y*y > halfPatch*halfPatch) continue; // Circular mask
                
                int px = cx + x;
                int py = cy + y;
                
                if (px < 0 || px >= image.cols || py < 0 || py >= image.rows) continue;
                
                uchar val = image.at<uchar>(py, px);
                m10 += x * val;
                m01 += y * val;
            }
        }
        
        float angle = cv::fastAtan2(m01, m10); // Returns degrees 0-360
        kp.angle = angle;
    }
}

cv::Mat extractOrbDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const OrbConfig& config) {
    initPattern(config.patchSize);
    
    cv::Mat descriptors = cv::Mat::zeros(keypoints.size(), 32, CV_8UC1);
    int halfPatch = config.patchSize / 2;
    
    // Smooth image for robustness
    cv::Mat smoothImg;
    cv::GaussianBlur(image, smoothImg, cv::Size(7, 7), 2.0);

    for (size_t i = 0; i < keypoints.size(); ++i) {
        const auto& kp = keypoints[i];
        float angleRad = kp.angle * CV_PI / 180.0f;
        float sina = std::sin(angleRad);
        float cosa = std::cos(angleRad);
        
        int cx = std::round(kp.pt.x);
        int cy = std::round(kp.pt.y);
        
        uchar* descRow = descriptors.ptr<uchar>(i);
        
        for (int b = 0; b < 256; ++b) {
            // Rotate pattern points
            cv::Point pA = patternA[b];
            cv::Point pB = patternB[b];
            
            // Rotate pA
            float ax_rot = pA.x * cosa - pA.y * sina;
            float ay_rot = pA.x * sina + pA.y * cosa;
            
            // Rotate pB
            float bx_rot = pB.x * cosa - pB.y * sina;
            float by_rot = pB.x * sina + pB.y * cosa;
            
            int ax = std::round(cx + ax_rot);
            int ay = std::round(cy + ay_rot);
            int bx = std::round(cx + bx_rot);
            int by = std::round(cy + by_rot);
            
            // Boundary check
            if (ax < 0 || ax >= smoothImg.cols || ay < 0 || ay >= smoothImg.rows ||
                bx < 0 || bx >= smoothImg.cols || by < 0 || by >= smoothImg.rows) {
                // Out of bounds, just set 0 or random. 
                // Usually ORB ignores keypoints near border.
                // Here we default to 0 (already zeroed).
                continue;
            }
            
            uchar valA = smoothImg.at<uchar>(ay, ax);
            uchar valB = smoothImg.at<uchar>(by, bx);
            
            if (valA < valB) {
                descRow[b / 8] |= (1 << (b % 8));
            }
        }
    }
    
    return descriptors;
}

} // namespace cv_curriculum
