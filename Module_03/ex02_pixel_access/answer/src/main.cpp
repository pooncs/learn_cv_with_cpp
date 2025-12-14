#include <iostream>
#include <chrono>
#include "pixel_access.hpp"

using namespace std::chrono;

int main() {
    // 4K image, 1 channel
    cv::Mat img = cv::Mat::zeros(2160, 3840, CV_8UC1);

    auto start = high_resolution_clock::now();
    process_at(img);
    auto stop = high_resolution_clock::now();
    std::cout << "at<>: " << duration_cast<milliseconds>(stop - start).count() << " ms\n";

    start = high_resolution_clock::now();
    process_ptr(img);
    stop = high_resolution_clock::now();
    std::cout << "ptr<>: " << duration_cast<milliseconds>(stop - start).count() << " ms\n";

    start = high_resolution_clock::now();
    process_iter(img);
    stop = high_resolution_clock::now();
    std::cout << "iter: " << duration_cast<milliseconds>(stop - start).count() << " ms\n";

    return 0;
}
