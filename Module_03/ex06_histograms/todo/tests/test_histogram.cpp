#include <iostream>
#include <cassert>
#include <numeric>
#include "histogram.hpp"

void test_hist() {
    cv::Mat img = cv::Mat::zeros(10, 10, CV_8UC1);
    img.at<uchar>(0,0) = 50;
    
    std::vector<int> hist = compute_histogram(img);
    // 99 zeros, 1 fifty
    assert(hist[0] == 99);
    assert(hist[50] == 1);
    assert(std::accumulate(hist.begin(), hist.end(), 0) == 100);
    
    std::cout << "[PASS] compute_histogram\n";
}

int main() {
    test_hist();
    return 0;
}
