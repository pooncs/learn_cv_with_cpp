#include <iostream>
#include <opencv2/opencv.hpp>
#include "triangulation.hpp"

int main(int argc, char** argv) {
    std::cout << "Triangulation Exercise" << std::endl;
    
    // Synthetic Setup
    // K = Identity, f = 1000, cx=cy=500
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0,0) = 1000;
    K.at<float>(1,1) = 1000;
    K.at<float>(0,2) = 500;
    K.at<float>(1,2) = 500;
    
    // Cam 1 at Origin
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32F);
    K.copyTo(P1(cv::Rect(0,0,3,3))); // P1 = K [I | 0]
    
    // Cam 2 translated by x=100
    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_32F);
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat t = (cv::Mat_<float>(3, 1) << -100, 0, 0); // Extrinsic: X_cam = R*X_world + t. 
    // If Cam 2 is at world (100,0,0), then in cam coords, world origin is at (-100,0,0).
    // P2 = K [R | t]
    
    cv::Mat Rt(3, 4, CV_32F);
    R.copyTo(Rt(cv::Rect(0,0,3,3)));
    t.copyTo(Rt(cv::Rect(3,0,1,3)));
    P2 = K * Rt;
    
    // Create a 3D point at (0, 0, 1000) (World)
    // In Cam 1: (0, 0, 1000). u = 1000*0/1000 + 500 = 500. v = 500.
    // In Cam 2: (-100, 0, 1000). u = 1000*-100/1000 + 500 = 400. v = 500.
    
    std::vector<cv::Point2f> pts1 = {{500, 500}};
    std::vector<cv::Point2f> pts2 = {{400, 500}};
    
    auto points3D = cv_curriculum::triangulateStereo(P1, P2, pts1, pts2);
    
    if (points3D.empty()) {
        std::cout << "Triangulation empty." << std::endl;
    } else {
        std::cout << "Reconstructed: " << points3D[0] << std::endl;
        std::cout << "Expected: [0, 0, 1000]" << std::endl;
    }
    
    return 0;
}
