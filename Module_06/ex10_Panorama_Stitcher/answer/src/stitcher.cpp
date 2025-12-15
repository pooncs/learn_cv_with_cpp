#include "stitcher.hpp"
#include <iostream>

namespace cv_curriculum {

cv::Mat stitchImages(const cv::Mat& img1, const cv::Mat& img2) {
    // 1. Detect & Compute
    auto orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    // 2. Match
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(desc2, desc1, knnMatches, 2); // Match img2 (query) to img1 (train)
    
    // 3. Filter
    std::vector<cv::DMatch> goodMatches;
    for (const auto& m : knnMatches) {
        if (m.size() >= 2 && m[0].distance < 0.75 * m[1].distance) {
            goodMatches.push_back(m[0]);
        }
    }
    
    if (goodMatches.size() < 4) {
        std::cerr << "Not enough matches to stitch." << std::endl;
        return cv::Mat();
    }
    
    // 4. Homography
    std::vector<cv::Point2f> pts2, pts1;
    for (const auto& m : goodMatches) {
        pts2.push_back(kp2[m.queryIdx].pt);
        pts1.push_back(kp1[m.trainIdx].pt);
    }
    
    cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC, 3.0);
    if (H.empty()) return cv::Mat();
    
    // 5. Warp & Blend
    // Compute size of result
    // Warp corners of img2 to see where they land
    std::vector<cv::Point2f> corners2(4);
    corners2[0] = cv::Point2f(0, 0);
    corners2[1] = cv::Point2f(img2.cols, 0);
    corners2[2] = cv::Point2f(img2.cols, img2.rows);
    corners2[3] = cv::Point2f(0, img2.rows);
    std::vector<cv::Point2f> warpedCorners2;
    cv::perspectiveTransform(corners2, warpedCorners2, H);
    
    // Combine with img1 corners
    std::vector<cv::Point2f> allCorners = warpedCorners2;
    allCorners.emplace_back(0, 0);
    allCorners.emplace_back(img1.cols, 0);
    allCorners.emplace_back(img1.cols, img1.rows);
    allCorners.emplace_back(0, img1.rows);
    
    cv::Rect boundingBox = cv::boundingRect(allCorners);
    
    // Create translation matrix to shift to positive coordinates
    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    if (boundingBox.x < 0) T.at<double>(0, 2) = -boundingBox.x;
    if (boundingBox.y < 0) T.at<double>(1, 2) = -boundingBox.y;
    
    cv::Mat H_final = T * H;
    
    cv::Mat panorama;
    cv::warpPerspective(img2, panorama, H_final, boundingBox.size());
    
    // Place img1
    cv::Mat roi(panorama, cv::Rect(-std::min(0, boundingBox.x), -std::min(0, boundingBox.y), img1.cols, img1.rows));
    img1.copyTo(roi);
    
    return panorama;
}

} // namespace cv_curriculum
