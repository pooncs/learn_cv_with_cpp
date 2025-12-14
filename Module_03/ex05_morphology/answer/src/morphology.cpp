#include "morphology.hpp"

cv::Mat my_dilate(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat dst = src.clone(); // Initialize with src or zeros? Better zeros for logic.
    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    for(int i=1; i<src.rows-1; ++i) {
        for(int j=1; j<src.cols-1; ++j) {
            uchar max_val = 0;
            for(int ki=-1; ki<=1; ++ki) {
                for(int kj=-1; kj<=1; ++kj) {
                    if(src.at<uchar>(i+ki, j+kj) == 255) {
                        max_val = 255;
                    }
                }
            }
            dst.at<uchar>(i, j) = max_val;
        }
    }
    return dst;
}

cv::Mat my_erode(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);

    for(int i=1; i<src.rows-1; ++i) {
        for(int j=1; j<src.cols-1; ++j) {
            uchar min_val = 255;
            for(int ki=-1; ki<=1; ++ki) {
                for(int kj=-1; kj<=1; ++kj) {
                    if(src.at<uchar>(i+ki, j+kj) == 0) {
                        min_val = 0;
                    }
                }
            }
            dst.at<uchar>(i, j) = min_val;
        }
    }
    return dst;
}
