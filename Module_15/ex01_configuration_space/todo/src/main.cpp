#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include <vector>

// --- Configuration ---
const int WIDTH = 800;
const int HEIGHT = 600;
const int ROBOT_RADIUS = 25;

// --- Helper: Draw Obstacles ---
void drawObstacles(cv::Mat& img, const std::vector<cv::Rect>& obstacles, const cv::Scalar& color) {
    // TODO: Iterate over obstacles and draw them on 'img' using cv::rectangle
}

// --- Helper: Generate C-Space ---
// Concept: C-Space Obstacle = Obstacle \oplus Robot
cv::Mat generateCSpace(const cv::Mat& workspace, int robotRadius) {
    cv::Mat cspace;
    // TODO: Implement Minkowski Sum equivalent using Morphological Dilation.
    // 1. Create a circular structuring element using cv::getStructuringElement
    // 2. Dilate the workspace image into 'cspace' using cv::dilate
    return cspace;
}

int main() {
    // 1. Setup Workspace
    cv::Mat workspace = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
    
    std::vector<cv::Rect> obstacles = {
        cv::Rect(200, 150, 100, 300),
        cv::Rect(500, 100, 150, 100),
        cv::Rect(450, 400, 200, 50)
    };

    // TODO: Call drawObstacles
    drawObstacles(workspace, obstacles, cv::Scalar(255));

    // 2. Generate C-Space
    // TODO: Call generateCSpace
    cv::Mat cspace = generateCSpace(workspace, ROBOT_RADIUS);

    // 3. Visualization Setup
    cv::Mat displayWorkspace, displayCSpace;
    cv::cvtColor(workspace, displayWorkspace, cv::COLOR_GRAY2BGR);
    
    // Check if cspace is empty before converting (in case student hasn't implemented it yet)
    if (!cspace.empty()) {
        cv::cvtColor(cspace, displayCSpace, cv::COLOR_GRAY2BGR);
    } else {
        displayCSpace = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    }

    fmt::print("Controls:\n  Click on images to place robot.\n  Press 'ESC' to exit.\n");

    // 4. Interactive Loop
    cv::namedWindow("Workspace", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("C-Space", cv::WINDOW_AUTOSIZE);

    auto onMouse = [&](int event, int x, int y, int flags, void* userdata) {
        if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_MOUSEMOVE) return;
        if (!(flags & cv::EVENT_FLAG_LBUTTON)) return;

        cv::Mat* targetImg = (cv::Mat*)userdata;
        cv::Mat show = targetImg->clone();

        // TODO: Check collision in C-Space
        // Hint: Check pixel value at (x, y) in 'cspace'
        bool collision = false; 

        cv::Scalar robotColor = collision ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        // Draw Robot
        cv::circle(show, cv::Point(x, y), ROBOT_RADIUS, robotColor, 2);
        
        const char* winName = (targetImg == &displayWorkspace) ? "Workspace" : "C-Space";
        cv::imshow(winName, show);
    };

    cv::setMouseCallback("Workspace", onMouse, &displayWorkspace);
    cv::setMouseCallback("C-Space", onMouse, &displayCSpace);

    cv::imshow("Workspace", displayWorkspace);
    cv::imshow("C-Space", displayCSpace);

    cv::waitKey(0);

    return 0;
}
