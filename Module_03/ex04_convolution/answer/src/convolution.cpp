#include "convolution.hpp"

cv::Mat convolve(const cv::Mat& src, const cv::Mat& kernel) {
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(kernel.type() == CV_32F);
    CV_Assert(kernel.rows == 3 && kernel.cols == 3);

    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);
    int pad = 1;

    for(int i=0; i<src.rows; ++i) {
        for(int j=0; j<src.cols; ++j) {
            float sum = 0.0f;
            
            // 3x3 window
            for(int ki=-1; ki<=1; ++ki) {
                for(int kj=-1; kj<=1; ++kj) {
                    int ni = i + ki;
                    int nj = j + kj;
                    
                    // Zero padding check
                    float val = 0.0f;
                    if(ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                        val = static_cast<float>(src.at<uchar>(ni, nj));
                    }
                    
                    // Kernel indices: 0..2
                    float k_val = kernel.at<float>(ki + 1, kj + 1);
                    sum += val * k_val;
                }
            }
            
            // Clamp to 0-255
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            
            dst.at<uchar>(i, j) = static_cast<uchar>(sum);
        }
    }
    return dst;
}
