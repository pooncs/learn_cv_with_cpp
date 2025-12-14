#include <iostream>
#include "gradients.hpp"

int main() {
    // Create a simple image with a vertical edge
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC1);
    // Right half white
    img(cv::Rect(50, 0, 50, 100)) = 255;

    auto [Gx, Gy] = compute_sobel(img);
    auto [mag, angle] = compute_magnitude_angle(Gx, Gy);

    // Check center pixel (49, 50) is the edge
    // Left is 0, Right is 255.
    // Sobel X kernel [-1 0 1]. So at edge: (255*1 + 0*-1) roughly positive
    // But Sobel is 3x3, so it smooths.
    
    // Visualize
    cv::Mat Gx_vis, Gy_vis, mag_vis;
    cv::convertScaleAbs(Gx, Gx_vis);
    cv::convertScaleAbs(Gy, Gy_vis);
    cv::convertScaleAbs(mag, mag_vis);

    std::cout << "Computed Gradients.\n";
    // cv::imshow("Gx", Gx_vis);
    // cv::imshow("Mag", mag_vis);
    // cv::waitKey(0);

    return 0;
}
