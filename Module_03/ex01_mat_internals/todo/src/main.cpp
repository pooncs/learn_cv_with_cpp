#include <iostream>
#include "mat_info.hpp"

int main() {
    // Task 1: Create Mat
    cv::Mat A(300, 400, CV_8UC3, cv::Scalar(0, 0, 255)); // Red image

    std::cout << "=== Task 1 & 2: Mat Info ===\n";
    std::cout << get_mat_info(A) << "\n";

    std::cout << "=== Task 3: Reference Counting ===\n";
    cv::Mat B = A; // Shallow copy
    cv::Mat C = A.clone(); // Deep copy

    std::cout << "A and B share data? " << (share_data(A, B) ? "Yes" : "No") << "\n";
    std::cout << "A and C share data? " << (share_data(A, C) ? "Yes" : "No") << "\n";

    return 0;
}
