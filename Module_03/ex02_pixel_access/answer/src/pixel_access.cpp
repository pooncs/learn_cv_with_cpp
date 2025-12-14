#include "pixel_access.hpp"

void process_at(cv::Mat& img) {
    for(int i=0; i<img.rows; ++i) {
        for(int j=0; j<img.cols; ++j) {
            img.at<uchar>(i, j) += 1;
        }
    }
}

void process_ptr(cv::Mat& img) {
    for(int i=0; i<img.rows; ++i) {
        uchar* row_ptr = img.ptr<uchar>(i);
        for(int j=0; j<img.cols; ++j) {
            row_ptr[j] += 1;
        }
    }
}

void process_iter(cv::Mat& img) {
    cv::MatIterator_<uchar> it, end;
    for(it = img.begin<uchar>(), end = img.end<uchar>(); it != end; ++it) {
        *it += 1;
    }
}
