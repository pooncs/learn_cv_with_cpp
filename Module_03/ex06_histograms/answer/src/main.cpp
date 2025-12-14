#include <iostream>
#include "histogram.hpp"

int main() {
    // Gradient image 0-255
    cv::Mat img(256, 256, CV_8UC1);
    for(int i=0; i<256; ++i) {
        for(int j=0; j<256; ++j) {
            img.at<uchar>(i, j) = (uchar)j;
        }
    }

    std::vector<int> hist = compute_histogram(img);
    cv::Mat hist_vis = draw_histogram(hist);
    
    cv::Mat eq = equalize_hist_manual(img);
    
    // Since input is already uniform, equalized should look similar.
    // Try creating a dark image.
    cv::Mat dark = img * 0.5;
    cv::Mat dark_eq = equalize_hist_manual(dark);

    std::cout << "Computed histograms and equalization.\n";
    // cv::imshow("Hist", hist_vis);
    // cv::waitKey(0);

    return 0;
}
