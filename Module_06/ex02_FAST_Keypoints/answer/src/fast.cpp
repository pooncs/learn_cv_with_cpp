#include "fast.hpp"
#include <cmath>

namespace cv_curriculum {

// Offsets for the 16 pixels in the Bresenham circle of radius 3
static const int offsets[16][2] = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1},
    {3, 0}, {3, 1}, {2, 2}, {1, 3},
    {0, 3}, {-1, 3}, {-2, 2}, {-3, 1},
    {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
};

int computeScore(const cv::Mat& img, int u, int v, int threshold, const std::vector<int>& flags) {
    // Score defined as sum of absolute difference between p and circle pixels
    // Only for pixels in the contiguous arc
    // But OpenCV uses the max threshold for which it is still a corner.
    // A simpler score for NMS: Sum of absolute differences (SAD) for the pixels that triggered the detection.
    int center = img.at<uchar>(v, u);
    int score = 0;
    // This is a simplified score: just sum of diffs
    for (int i = 0; i < 16; ++i) {
         int val = img.at<uchar>(v + offsets[i][1], u + offsets[i][0]);
         if (std::abs(val - center) > threshold) {
             score += std::abs(val - center) - threshold;
         }
    }
    return score;
}

std::vector<cv::KeyPoint> detectFAST(const cv::Mat& gray, const FastConfig& config) {
    std::vector<cv::KeyPoint> keypoints;
    if (gray.empty()) return keypoints;

    int t = config.threshold;
    int N = config.N;

    // Temporary map to store scores for NMS
    cv::Mat scores = cv::Mat::zeros(gray.size(), CV_32S);
    std::vector<cv::Point> corners;

    for (int y = 3; y < gray.rows - 3; ++y) {
        for (int x = 3; x < gray.cols - 3; ++x) {
            uchar p = gray.at<uchar>(y, x);
            
            // Optimization: Check indices 0, 4, 8, 12
            // At least 3 of these must be significantly different for a corner to exist (assuming N >= 9)
            // But strict check depends on N.
            // Let's implement the full check for correctness first, optimization second.
            
            // Collect pixels
            int circle[16];
            for (int k = 0; k < 16; ++k) {
                circle[k] = gray.at<uchar>(y + offsets[k][1], x + offsets[k][0]);
            }

            // Check for N contiguous brighter
            bool found = false;
            
            // Brighter loop
            for (int k = 0; k < 16; ++k) {
                int count = 0;
                for (int i = 0; i < N; ++i) {
                    int idx = (k + i) % 16;
                    if (circle[idx] > p + t) count++;
                    else break;
                }
                if (count == N) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                // Darker loop
                for (int k = 0; k < 16; ++k) {
                    int count = 0;
                    for (int i = 0; i < N; ++i) {
                        int idx = (k + i) % 16;
                        if (circle[idx] < p - t) count++;
                        else break;
                    }
                    if (count == N) {
                        found = true;
                        break;
                    }
                }
            }

            if (found) {
                int score = computeScore(gray, x, y, t, {}); // Simplified score
                scores.at<int>(y, x) = score;
                corners.emplace_back(x, y);
            }
        }
    }

    // NMS
    if (config.nonmaxSuppression) {
        for (const auto& pt : corners) {
            int x = pt.x;
            int y = pt.y;
            int score = scores.at<int>(y, x);
            
            bool isMax = true;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (scores.at<int>(y + dy, x + dx) >= score) {
                        isMax = false; // If equal, prefer one (doesn't matter which, but strict > helps reduce clusters)
                        // Actually if equal, we might suppress both if we use >, or keep both if >=. 
                        // Standard NMS usually suppresses if neighbor is greater.
                        if (scores.at<int>(y + dy, x + dx) > score) isMax = false;
                    }
                }
            }
            if (isMax) {
                keypoints.emplace_back(cv::Point2f((float)x, (float)y), (float)3.0, -1, (float)score);
            }
        }
    } else {
        for (const auto& pt : corners) {
             keypoints.emplace_back(cv::Point2f((float)pt.x, (float)pt.y), (float)3.0, -1, (float)scores.at<int>(pt.y, pt.x));
        }
    }

    return keypoints;
}

} // namespace cv_curriculum
