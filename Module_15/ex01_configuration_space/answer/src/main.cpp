#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include <vector>

// --- Configuration ---
const int WIDTH = 800;
const int HEIGHT = 600;
const int ROBOT_RADIUS = 25;

// --- Helper: Draw Obstacles ---
void drawObstacles(cv::Mat& img, const std::vector<cv::Rect>& obstacles, const cv::Scalar& color) {
    for (const auto& obs : obstacles) {
        cv::rectangle(img, obs, color, -1);
    }
}

// --- Helper: Generate C-Space ---
// Concept: C-Space Obstacle = Obstacle \oplus Robot
// For a circular robot, this is equivalent to dilating the obstacle map with a circular structuring element.
cv::Mat generateCSpace(const cv::Mat& workspace, int robotRadius) {
    cv::Mat cspace;
    // Create a circular structuring element
    int kernelSize = 2 * robotRadius + 1; // Must be odd
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
    
    // Dilate the workspace (white obstacles) to inflate them
    cv::dilate(workspace, cspace, element);
    return cspace;
}

// --- Test Function ---
void runTests(const cv::Mat& cspace, const std::vector<cv::Rect>& obstacles) {
    fmt::print("Running verification tests...\n");

    // Test 1: Point deep inside obstacle should be collision in C-Space
    cv::Point pInside = obstacles[0].tl() + cv::Point(10, 10);
    if (cspace.at<uchar>(pInside) == 255) {
        fmt::print("[PASS] Point inside obstacle is occupied in C-Space.\n");
    } else {
        fmt::print("[FAIL] Point inside obstacle should be occupied!\n");
    }

    // Test 2: Point far away should be free
    cv::Point pFree(10, 10); // Assuming (10,10) is empty
    if (cspace.at<uchar>(pFree) == 0) {
        fmt::print("[PASS] Free space point is free in C-Space.\n");
    } else {
        fmt::print("[FAIL] Free space point should be free!\n");
    }
    
    // Test 3: Point just outside obstacle but within radius (Collision)
    // Right of the first obstacle
    cv::Rect obs = obstacles[0];
    cv::Point pNear = cv::Point(obs.x + obs.width + ROBOT_RADIUS - 1, obs.y + obs.height/2);
    
    // Check bounds
    if (pNear.x < WIDTH && pNear.y < HEIGHT) {
        if (cspace.at<uchar>(pNear) == 255) {
            fmt::print("[PASS] Point within radius distance is occupied in C-Space (Correct).\n");
        } else {
            fmt::print("[FAIL] Point within radius distance should be occupied!\n");
        }
    }
}

int main() {
    // 1. Setup Workspace
    cv::Mat workspace = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
    
    std::vector<cv::Rect> obstacles = {
        cv::Rect(200, 150, 100, 300),
        cv::Rect(500, 100, 150, 100),
        cv::Rect(450, 400, 200, 50)
    };

    drawObstacles(workspace, obstacles, cv::Scalar(255));

    // 2. Generate C-Space
    cv::Mat cspace = generateCSpace(workspace, ROBOT_RADIUS);

    // 3. Visualization Setup
    cv::Mat displayWorkspace, displayCSpace;
    cv::cvtColor(workspace, displayWorkspace, cv::COLOR_GRAY2BGR);
    cv::cvtColor(cspace, displayCSpace, cv::COLOR_GRAY2BGR);

    // Overlay "Ghost" of original obstacles on C-Space for comparison
    // We draw the original obstacles in gray on the C-Space map to show the inflation
    for(const auto& obs : obstacles) {
        cv::rectangle(displayCSpace, obs, cv::Scalar(100, 100, 100), 1);
    }

    // Run verification
    runTests(cspace, obstacles);

    fmt::print("Controls:\n  Click on images to place robot.\n  Press 'ESC' to exit.\n");

    // 4. Interactive Loop
    cv::namedWindow("Workspace", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("C-Space", cv::WINDOW_AUTOSIZE);

    // Mouse callback lambda
    auto onMouse = [&](int event, int x, int y, int flags, void* userdata) {
        if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_MOUSEMOVE) return;
        if (!(flags & cv::EVENT_FLAG_LBUTTON)) return;

        cv::Mat* targetImg = (cv::Mat*)userdata;
        cv::Mat show = targetImg->clone();

        // Check collision in C-Space
        // Note: We check the C-Space map regardless of which window we click in for the logic,
        // but let's just visualize based on where we are.
        
        bool collision = false;
        if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
             collision = cspace.at<uchar>(y, x) > 0;
        }

        cv::Scalar robotColor = collision ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        // Draw Robot
        // In Workspace: It's a circle with radius
        cv::circle(show, cv::Point(x, y), ROBOT_RADIUS, robotColor, 2);
        
        // Draw Center
        cv::circle(show, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), -1);

        const char* winName = (targetImg == &displayWorkspace) ? "Workspace" : "C-Space";
        
        std::string status = collision ? "COLLISION" : "FREE";
        cv::putText(show, status, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, robotColor, 2);

        cv::imshow(winName, show);
    };

    cv::setMouseCallback("Workspace", onMouse, &displayWorkspace);
    cv::setMouseCallback("C-Space", onMouse, &displayCSpace);

    // Initial show
    cv::imshow("Workspace", displayWorkspace);
    cv::imshow("C-Space", displayCSpace);

    cv::waitKey(0);

    return 0;
}
