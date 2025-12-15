#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "flann_matcher.hpp"

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
        img1 = cv::Mat::zeros(800, 800, CV_8UC1); // Larger image
        cv::randn(img1, 128, 50);
    }
    
    cv::Mat img2;
    cv::Point2f center(img1.cols/2.0f, img1.rows/2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, 15, 1.0);
    cv::warpAffine(img1, img2, rot, img1.size());
    
    // Use ORB with many points
    auto orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    std::cout << "Features: " << desc1.rows << " vs " << desc2.rows << std::endl;
    
    // 1. FLANN Match
    auto flann = cv_curriculum::createFlannLshMatcher();
    
    auto start = std::chrono::high_resolution_clock::now();
    auto matchesFlann = cv_curriculum::matchFlann(flann, desc1, desc2);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> dt = end - start;
    std::cout << "FLANN Matches: " << matchesFlann.size() << " in " << dt.count() << " ms" << std::endl;
    
    // 2. BF Match for comparison
    auto bf = cv::BFMatcher::create(cv::NORM_HAMMING);
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<cv::DMatch>> knnMatches;
    bf->knnMatch(desc1, desc2, knnMatches, 2);
    // Ratio filter loop...
    int goodCount = 0;
    for (auto& m : knnMatches) {
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance) goodCount++;
    }
    end = std::chrono::high_resolution_clock::now();
    dt = end - start;
    
    std::cout << "BF Matches: " << goodCount << " in " << dt.count() << " ms" << std::endl;
    
    cv::Mat outImg;
    cv::drawMatches(img1, kp1, img2, kp2, matchesFlann, outImg);
    cv::imwrite("flann_matches.png", outImg);

    return 0;
}
