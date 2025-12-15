#include <iostream>
#include <opencv2/opencv.hpp>
#include "matcher.hpp"

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

    cv::Mat img1 = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img1.empty()) {
        std::cout << "Image not found at " << imagePath << ", creating synthetic." << std::endl;
        img1 = cv::Mat::zeros(400, 400, CV_8UC1);
        cv::randn(img1, 128, 50); // Random noise texture
    }

    // Create img2 by rotating img1
    cv::Mat img2;
    cv::Point2f center(img1.cols/2.0f, img1.rows/2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, 15, 1.0);
    cv::warpAffine(img1, img2, rot, img1.size());
    
    // Detect and Compute ORB
    auto orb = cv::ORB::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    std::cout << "Features 1: " << kp1.size() << std::endl;
    std::cout << "Features 2: " << kp2.size() << std::endl;
    
    // Match
    auto matches = cv_curriculum::matchBruteForce(desc1, desc2);
    
    std::cout << "Matches found: " << matches.size() << std::endl;
    
    // Draw matches
    cv::Mat imgMatches;
    cv::drawMatches(img1, kp1, img2, kp2, matches, imgMatches);
    
    cv::imwrite("bf_matches.png", imgMatches);
    
    return 0;
}
