#include <iostream>
#include "pnp.hpp"

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);

    // 3D Points (Square on Z=0)
    std::vector<cv::Point3f> obj_pts = {
        {0, 0, 0}, {10, 0, 0}, {10, 10, 0}, {0, 10, 0}
    };

    // 2D Points (Simulated projection with some T)
    std::vector<cv::Point2f> img_pts = {
        {320, 240}, {820, 240}, {820, 740}, {320, 740}
    };

    // TODO: Call estimate_pose

    return 0;
}
