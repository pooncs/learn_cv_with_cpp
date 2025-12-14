#include "color_spaces.hpp"
#include <algorithm>
#include <cmath>

cv::Mat to_gray(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8UC3);
    cv::Mat gray(src.rows, src.cols, CV_8UC1);

    for(int i=0; i<src.rows; ++i) {
        const cv::Vec3b* row_ptr = src.ptr<cv::Vec3b>(i);
        uchar* gray_ptr = gray.ptr<uchar>(i);
        for(int j=0; j<src.cols; ++j) {
            uchar b = row_ptr[j][0];
            uchar g = row_ptr[j][1];
            uchar r = row_ptr[j][2];
            // Y = 0.299R + 0.587G + 0.114B
            gray_ptr[j] = static_cast<uchar>(0.299*r + 0.587*g + 0.114*b);
        }
    }
    return gray;
}

cv::Mat to_hsv(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8UC3);
    cv::Mat hsv(src.rows, src.cols, CV_8UC3);

    for(int i=0; i<src.rows; ++i) {
        const cv::Vec3b* row_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* hsv_ptr = hsv.ptr<cv::Vec3b>(i);
        for(int j=0; j<src.cols; ++j) {
            float b = row_ptr[j][0] / 255.0f;
            float g = row_ptr[j][1] / 255.0f;
            float r = row_ptr[j][2] / 255.0f;

            float v = std::max({r, g, b});
            float m = std::min({r, g, b});
            float c = v - m;

            float s = (v > 0) ? (c / v) : 0;
            float h = 0;

            if (c > 0) {
                if (v == r) h = (g - b) / c;
                else if (v == g) h = 2.0f + (b - r) / c;
                else h = 4.0f + (r - g) / c;
                
                h *= 60.0f;
                if (h < 0) h += 360.0f;
            }
            
            // Map to OpenCV ranges: H [0,180], S [0,255], V [0,255]
            hsv_ptr[j][0] = static_cast<uchar>(h / 2.0f);
            hsv_ptr[j][1] = static_cast<uchar>(s * 255.0f);
            hsv_ptr[j][2] = static_cast<uchar>(v * 255.0f);
        }
    }
    return hsv;
}
