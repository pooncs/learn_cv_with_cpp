#include "thresholding.hpp"

cv::Mat my_threshold(const cv::Mat& src, int thresh) {
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat dst(src.size(), CV_8UC1);
    for(int i=0; i<src.rows; ++i) {
        const uchar* s = src.ptr<uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for(int j=0; j<src.cols; ++j) {
            d[j] = (s[j] > thresh) ? 255 : 0;
        }
    }
    return dst;
}

cv::Mat my_adaptive_threshold(const cv::Mat& src, int blockSize, int C) {
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat mean;
    // Calculate local mean using box filter
    cv::blur(src, mean, cv::Size(blockSize, blockSize));
    
    cv::Mat dst(src.size(), CV_8UC1);
    for(int i=0; i<src.rows; ++i) {
        const uchar* s = src.ptr<uchar>(i);
        const uchar* m = mean.ptr<uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for(int j=0; j<src.cols; ++j) {
            int thresh = m[j] - C;
            d[j] = (s[j] > thresh) ? 255 : 0;
        }
    }
    return dst;
}
