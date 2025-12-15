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
    // T = [0, 0, 20]
    // p = f * X/Z + c
    // (0,0,0) -> (0,0,20) -> (320, 240)
    // (10,0,0) -> (10,0,20) -> (1000*0.5+320, 240) = (820, 240)
    std::vector<cv::Point2f> img_pts = {
        {320, 240}, {820, 240}, {820, 740}, {320, 740}
    };

    auto [rvec, tvec] = estimate_pose(obj_pts, img_pts, K, dist);

    std::cout << "tvec: " << tvec.t() << "\n"; // Should be close to [0, 0, 20]

    cv::Mat img = cv::Mat::zeros(1000, 1200, CV_8UC3);
    draw_axes(img, K, dist, rvec, tvec, 5.0f);
    
    // cv::imshow("Pose", img);
    // cv::waitKey(0);

    return 0;
}
