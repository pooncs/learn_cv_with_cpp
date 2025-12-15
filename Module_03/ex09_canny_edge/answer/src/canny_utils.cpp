#include "canny_utils.hpp"
#include <cmath>

cv::Mat non_max_suppression(const cv::Mat& mag, const cv::Mat& angle) {
    CV_Assert(mag.type() == CV_32F && angle.type() == CV_32F);
    cv::Mat nms = cv::Mat::zeros(mag.size(), CV_32F);

    int rows = mag.rows;
    int cols = mag.cols;

    for(int i=1; i<rows-1; ++i) {
        for(int j=1; j<cols-1; ++j) {
            float ang = angle.at<float>(i, j);
            float val = mag.at<float>(i, j);
            
            // Normalize angle to [0, 180)
            if (ang < 0) ang += 180;
            if (ang >= 180) ang -= 180;

            float q = 255;
            float r = 255;

            // 0 degrees (Horizontal) -> Check Left/Right
            if ((0 <= ang && ang < 22.5) || (157.5 <= ang && ang <= 180)) {
                q = mag.at<float>(i, j+1);
                r = mag.at<float>(i, j-1);
            }
            // 45 degrees -> TopRight / BottomLeft
            else if (22.5 <= ang && ang < 67.5) {
                q = mag.at<float>(i+1, j-1); // Bottom-Left (y+1, x-1)
                r = mag.at<float>(i-1, j+1); // Top-Right (y-1, x+1)
            }
            // 90 degrees (Vertical) -> Top / Bottom
            else if (67.5 <= ang && ang < 112.5) {
                q = mag.at<float>(i+1, j);
                r = mag.at<float>(i-1, j);
            }
            // 135 degrees -> TopLeft / BottomRight
            else if (112.5 <= ang && ang < 157.5) {
                q = mag.at<float>(i-1, j-1); // Top-Left
                r = mag.at<float>(i+1, j+1); // Bottom-Right
            }

            if (val >= q && val >= r) {
                nms.at<float>(i, j) = val;
            } else {
                nms.at<float>(i, j) = 0;
            }
        }
    }
    
    // Convert to 8U for display
    cv::Mat res;
    nms.convertTo(res, CV_8U);
    return res;
}
