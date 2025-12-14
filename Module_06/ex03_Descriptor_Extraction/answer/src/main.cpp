#include <iostream>
#include <opencv2/opencv.hpp>
#include "orb_custom.hpp"

int main(int argc, char** argv) {
    const cv::String keys =
        "{help h usage ? |      | print this message   }"
        "{@image         |      | image for processing }";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    std::string imagePath = parser.get<std::string>("@image");
    if (imagePath.empty()) {
        imagePath = "../data/checkerboard.png"; 
    }

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
         std::cout << "Image not found at " << imagePath << ", creating synthetic image." << std::endl;
        img = cv::Mat::zeros(400, 400, CV_8UC3);
        // Draw rotated rectangle
        cv::Point center(200, 200);
        cv::Size size(100, 50);
        float angle = 30.0f;
        cv::RotatedRect rRect(center, size, angle);
        cv::Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            cv::line(img, vertices[i], vertices[(i+1)%4], cv::Scalar(255,255,255), -1);
    }
    
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();
    
    // 1. Detect Keypoints
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(gray, keypoints, 20, true);
    
    // 2. Compute Orientation
    cv_curriculum::computeOrientation(gray, keypoints);
    
    // 3. Extract Descriptors
    cv_curriculum::OrbConfig config;
    cv::Mat descriptors = cv_curriculum::extractOrbDescriptors(gray, keypoints, config);
    
    std::cout << "Extracted " << descriptors.rows << " descriptors." << std::endl;
    std::cout << "Descriptor type: " << descriptors.type() << " (Expected " << CV_8U << ")" << std::endl;
    std::cout << "Descriptor cols: " << descriptors.cols << " (Expected 32)" << std::endl;
    
    // Draw keypoints with orientation
    cv::Mat outImg;
    cv::drawKeypoints(img, keypoints, outImg, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imwrite("orb_result.png", outImg);
    
    return 0;
}
