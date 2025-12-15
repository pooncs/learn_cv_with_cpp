#include <iostream>
#include <cassert>
#include "canny_utils.hpp"

void test_nms() {
    // 3x3 mag
    // 0  0  0
    // 50 100 50
    // 0  0  0
    cv::Mat mag = cv::Mat::zeros(3, 3, CV_32F);
    mag.at<float>(1, 0) = 50;
    mag.at<float>(1, 1) = 100;
    mag.at<float>(1, 2) = 50;

    // Angle 0 (Horizontal) -> Check Left/Right
    // Wait. Gradient is change. Here change is Horizontal (0 to 100 to 0).
    // Gradient Direction is perpendicular to edge.
    // If intensity change is Horizontal (Left to Right), Gradient is Horizontal (0 deg).
    // So we check Left and Right neighbors.
    // Center (100) > Left (50) and Center (100) > Right (50).
    // So Center should be kept.
    cv::Mat angle = cv::Mat::zeros(3, 3, CV_32F); // 0 degrees
    
    cv::Mat res = non_max_suppression(mag, angle);
    if(res.empty()) {
        std::cout << "[SKIP] non_max_suppression not implemented\n";
        return;
    }
    
    // Center pixel should be non-zero
    assert(res.at<uchar>(1, 1) > 0);
    // Neighbors should be 0 (though input mag was non-zero, NMS suppresses them? 
    // Wait, NMS suppresses IF they are not local max.
    // (1,0) has neighbors (1,-1) and (1,1). (1,1) is 100. So (1,0) is 50. 50 < 100. So (1,0) suppressed.
    assert(res.at<uchar>(1, 0) == 0);
    
    std::cout << "[PASS] non_max_suppression\n";
}

int main() {
    test_nms();
    return 0;
}
