#include <iostream>
#include <opencv2/opencv.hpp>
#include "robust_matcher.hpp"

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
        cv::line(img1, cv::Point(0,0), cv::Point(400,400), cv::Scalar(255), 5);
        cv::rectangle(img1, cv::Rect(50, 50, 100, 100), cv::Scalar(255), -1);
    }

    cv::Mat img2;
    // Perspective warp for ground truth
    cv::Point2f src[4] = {{0,0}, {400,0}, {400,400}, {0,400}};
    cv::Point2f dst[4] = {{50,50}, {350,20}, {380,380}, {20,350}};
    cv::Mat H_gt = cv::getPerspectiveTransform(src, dst);
    cv::warpPerspective(img1, img2, H_gt, img1.size());
    
    // Detect and Match
    auto orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true); // Cross-check
    std::vector<cv::DMatch> matches;
    matcher->match(desc1, desc2, matches);
    
    std::cout << "Raw Matches: " << matches.size() << std::endl;
    
    // Convert KeyPoints to Point2f
    std::vector<cv::Point2f> pts1, pts2;
    cv::KeyPoint::convert(kp1, pts1);
    cv::KeyPoint::convert(kp2, pts2);
    
    // Robust Homography
    auto result = cv_curriculum::computeRobustHomography(pts1, pts2, matches, 3.0);
    
    std::cout << "Inliers: " << result.inlierMatches.size() << std::endl;
    if (!result.H.empty()) {
        std::cout << "Homography found:\n" << result.H << std::endl;
        
        // Warp img1 to align with img2
        cv::Mat warped;
        cv::warpPerspective(img1, warped, result.H, img1.size());
        cv::imwrite("warped_result.png", warped);
    }
    
    cv::Mat imgMatches;
    cv::drawMatches(img1, kp1, img2, kp2, result.inlierMatches, imgMatches);
    cv::imwrite("ransac_matches.png", imgMatches);
    
    return 0;
}
