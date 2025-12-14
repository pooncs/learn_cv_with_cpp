#include <iostream>
#include "gradients.hpp"

int main() {
    // Create a simple image with a vertical edge
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC1);
    // Right half white
    img(cv::Rect(50, 0, 50, 100)) = 255;

    auto [Gx, Gy] = compute_sobel(img);
    auto [mag, angle] = compute_magnitude_angle(Gx, Gy);

    std::cout << "Computed Gradients.\n";

    return 0;
}
