#include "histogram.hpp"
#include <algorithm>
#include <cmath>

std::vector<int> compute_histogram(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8UC1);
    std::vector<int> hist(256, 0);
    for(int i=0; i<src.rows; ++i) {
        const uchar* ptr = src.ptr<uchar>(i);
        for(int j=0; j<src.cols; ++j) {
            hist[ptr[j]]++;
        }
    }
    return hist;
}

cv::Mat draw_histogram(const std::vector<int>& hist, int width, int height) {
    cv::Mat hist_img = cv::Mat::zeros(height, width, CV_8UC1);
    int max_val = *std::max_element(hist.begin(), hist.end());
    if (max_val == 0) return hist_img;

    int bin_w = cvRound((double)width / 256);

    for(int i=0; i<256; ++i) {
        int h = cvRound(hist[i] * (double)height / max_val);
        cv::rectangle(hist_img, 
            cv::Point(i*bin_w, height), 
            cv::Point((i+1)*bin_w, height - h), 
            cv::Scalar(255), 
            cv::FILLED);
    }
    return hist_img;
}

cv::Mat equalize_hist_manual(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8UC1);
    std::vector<int> hist = compute_histogram(src);
    
    // CDF
    std::vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    for(int i=1; i<256; ++i) {
        cdf[i] = cdf[i-1] + hist[i];
    }

    // Normalize
    // h(v) = round( (cdf(v) - cdf_min) / (total_pixels - cdf_min) * 255 )
    // Simplified: h(v) = round( cdf(v) / total * 255 )
    int total = src.rows * src.cols;
    cv::Mat dst = src.clone();
    
    for(int i=0; i<src.rows; ++i) {
        uchar* ptr = dst.ptr<uchar>(i);
        const uchar* src_ptr = src.ptr<uchar>(i);
        for(int j=0; j<src.cols; ++j) {
            ptr[j] = cv::saturate_cast<uchar>(255.0 * cdf[src_ptr[j]] / total);
        }
    }
    return dst;
}
