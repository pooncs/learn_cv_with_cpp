#include <iostream>
#include <opencv2/opencv.hpp>
#include "harris.hpp"

int main(int argc, char** argv) {
    // Basic argument parsing
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
        cv::rectangle(img, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 255, 255), -1);
    }
    
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();
    
    cv_curriculum::HarrisConfig config;
    config.blockSize = 2;
    config.k = 0.04;
    
    // Compute Response
    cv::Mat response = cv_curriculum::computeHarrisResponse(gray, config);
    
    // Normalize for visualization
    cv::Mat responseNorm;
    cv::normalize(response, responseNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    // Detect Corners
    // Heuristic threshold: max * 0.01
    double minVal, maxVal;
    cv::minMaxLoc(response, &minVal, &maxVal);
    float threshold = static_cast<float>(maxVal * 0.01);
    
    auto corners = cv_curriculum::detectCorners(response, threshold);
    
    std::cout << "Detected " << corners.size() << " corners." << std::endl;
    
    // Draw
    for (const auto& pt : corners) {
        cv::circle(img, pt, 5, cv::Scalar(0, 0, 255), 2);
    }
    
    // Show only if GUI is available, otherwise save
    // We can assume GUI might not be available in some CI envs, but for user it is.
    // Let's just save the result too.
    cv::imwrite("harris_result.png", img);
    
    // cv::imshow("Harris Response", responseNorm);
    // cv::imshow("Corners", img);
    // cv::waitKey(0);
    
    return 0;
}
