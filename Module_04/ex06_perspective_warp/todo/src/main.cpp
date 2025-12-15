#include <iostream>
#include "warp.hpp"

int main() {
    cv::Mat img = cv::Mat::zeros(600, 800, CV_8UC3);
    
    // Draw a rotated document
    std::vector<cv::Point> doc = {
        {200, 150}, {500, 100}, {600, 400}, {300, 500}
    };
    cv::fillConvexPoly(img, doc, cv::Scalar(255, 255, 255));
    
    std::vector<cv::Point2f> corners;
    for(auto p : doc) corners.push_back(cv::Point2f(p.x, p.y));

    // TODO: Call rectify_document

    return 0;
}
