#include <iostream>
#include <opencv2/opencv.hpp>
#include "ratio_test.hpp"

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
        img1 = cv::Mat::zeros(400, 400, CV_8UC1);
        cv::randn(img1, 128, 50);
    }
    
    cv::Mat img2;
    cv::Point2f center(img1.cols/2.0f, img1.rows/2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, 10, 1.0);
    cv::warpAffine(img1, img2, rot, img1.size());
    
    auto orb = cv::ORB::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    std::cout << "Desc1: " << desc1.rows << ", Desc2: " << desc2.rows << std::endl;
    
    // KNN Match
    auto knnMatches = cv_curriculum::matchKnnBruteForce(desc1, desc2, 2);
    
    // Ratio Test
    auto goodMatches = cv_curriculum::filterRatioTest(knnMatches, 0.75f);
    
    std::cout << "Total KNN Matches: " << knnMatches.size() << std::endl;
    std::cout << "Good Matches (Ratio Test): " << goodMatches.size() << std::endl;
    
    cv::Mat imgMatches;
    cv::drawMatches(img1, kp1, img2, kp2, goodMatches, imgMatches);
    cv::imwrite("ratio_matches.png", imgMatches);
    
    return 0;
}
