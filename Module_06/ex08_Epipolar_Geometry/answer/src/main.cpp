#include <iostream>
#include <opencv2/opencv.hpp>
#include "epipolar.hpp"

int main(int argc, char** argv) {
    const cv::String keys =
        "{help h usage ? |      | print this message   }"
        "{@image         |      | image for processing }";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    
    // Create synthetic stereo scene (easier than finding stereo pair in default data)
    // Or just use two random views of a cube
    // Let's use the checkerboard again but warped
    std::string imagePath = parser.get<std::string>("@image");
    if (imagePath.empty()) {
        imagePath = "../data/checkerboard.png"; 
    }

    cv::Mat img1 = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img1.empty()) {
        img1 = cv::Mat::zeros(400, 400, CV_8UC1);
        cv::randn(img1, 128, 50);
        cv::rectangle(img1, cv::Rect(100,100,200,200), cv::Scalar(255), -1);
    }
    
    cv::Mat img2;
    // Simulate slight rotation and translation
    // Actually for F matrix, we need translation. If just rotation, F is undefined (use H).
    // Let's assume the scene is 3D (noise makes it pseudo-3D) or warp it in a way that mimics view change.
    // For this exercise, simple affine warp is "close enough" to test the pipeline, 
    // even if physically it corresponds to a planar scene (where F is degenerate).
    // BUT findFundamentalMat with 7-point or 8-point handles planar somewhat, though RANSAC might prefer H.
    // To be safe, let's use two completely different random noise images? No, need matches.
    // Let's use a synthetic set of points.
    
    // Synthetic Points Test Mode
    std::vector<cv::Point2f> pts1, pts2;
    int nPoints = 50;
    for(int i=0; i<nPoints; ++i) {
        pts1.emplace_back(rand() % 400, rand() % 400);
        // Pure translation: x' = x - 20
        pts2.emplace_back(pts1.back().x - 20 + (rand()%3 - 1), pts1.back().y + (rand()%3 - 1));
    }
    
    // Add outliers
    for(int i=0; i<10; ++i) {
        pts1.emplace_back(rand() % 400, rand() % 400);
        pts2.emplace_back(rand() % 400, rand() % 400);
    }
    
    auto result = cv_curriculum::computeFundamentalMatrix(pts1, pts2);
    std::cout << "Inliers: " << result.inliers1.size() << " / " << pts1.size() << std::endl;
    std::cout << "F:\n" << result.F << std::endl;
    
    // Compute Epilines
    std::vector<cv::Vec3f> lines1, lines2;
    cv::computeCorrespondEpilines(result.inliers1, 1, result.F, lines1);
    cv::computeCorrespondEpilines(result.inliers2, 2, result.F, lines2);
    
    // Draw
    cv::Mat outImg1 = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::Mat outImg2 = cv::Mat::zeros(400, 400, CV_8UC3);
    
    // We only have points, so draw on black canvas
    cv_curriculum::drawEpipolarLines(outImg1, lines2, result.inliers1); // Lines in 1 from points in 2? No.
    // computeCorrespondEpilines(points_in_image_1, 1, F, lines_in_image_2)
    // So lines1 corresponds to inliers1? No.
    // OpenCV docs: "computes epilines for the points in the other image".
    // whichImage=1 means points are in image 1. Lines are returned for image 2.
    // So lines1 (variable name) should be drawn on img2.
    
    cv_curriculum::drawEpipolarLines(outImg2, lines1, result.inliers2);
    cv_curriculum::drawEpipolarLines(outImg1, lines2, result.inliers1);
    
    cv::imwrite("epilines1.png", outImg1);
    cv::imwrite("epilines2.png", outImg2);
    
    return 0;
}
