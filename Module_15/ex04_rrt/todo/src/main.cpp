#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include <vector>
#include <cmath>
#include <random>

// --- Configuration ---
const int WIDTH = 800;
const int HEIGHT = 600;
const int STEP_SIZE = 30;
const int GOAL_THRESHOLD = 30;
const int MAX_ITER = 5000;

struct Point {
    float x, y;
};

struct Node {
    Point pos;
    int parent_idx; // Index in the tree vector
};

// --- Helper: Distance ---
float dist(Point a, Point b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// --- RRT Class ---
class RRT {
public:
    RRT(Point start, Point goal, const cv::Mat& map) 
        : start_(start), goal_(goal), map_(map) {
        nodes_.push_back({start, -1});
        
        // Random number generator setup
        std::random_device rd;
        gen_ = std::mt19937(rd());
        x_dist_ = std::uniform_real_distribution<>(0, WIDTH);
        y_dist_ = std::uniform_real_distribution<>(0, HEIGHT);
    }

    // Step 1: Sample
    Point getRandomPoint() {
        // TODO: Return a random point within [0, WIDTH] and [0, HEIGHT].
        // Optional: With 10% probability, return goal_ to bias search.
        return {0, 0};
    }

    // Step 2: Nearest
    int getNearestNodeIdx(Point query) {
        // TODO: Iterate through nodes_ and find the index of the node closest to 'query'.
        return 0;
    }

    // Step 3: Steer
    Point steer(Point from, Point to) {
        // TODO: Calculate a point that is 'STEP_SIZE' distance away from 'from' in the direction of 'to'.
        // If distance(from, to) < STEP_SIZE, return 'to'.
        return from;
    }

    // Step 4: Collision Check (Image based)
    bool isCollisionFree(Point from, Point to) {
        // TODO: Check if the line segment from 'from' to 'to' intersects any obstacle in 'map_'.
        // Hint: Sample points along the line and check map_.at<uchar>(y, x).
        // Return true if free, false if collision.
        return true;
    }

    // Main Loop Step
    bool step(cv::Mat& debug_img) {
        // TODO: Implement the RRT logic:
        // 1. rand_pt = getRandomPoint()
        // 2. nearest_idx = getNearestNodeIdx(rand_pt)
        // 3. new_pt = steer(nodes_[nearest_idx].pos, rand_pt)
        // 4. if (isCollisionFree(...)) {
        //      Add new node to nodes_
        //      Visualize
        //      Check if goal reached
        //    }
        return false;
    }

    std::vector<Point> getPath() {
        if (nodes_.empty()) return {};
        // Basic path reconstruction (assuming last node is goal)
        // This logic needs to be robust in the actual implementation
        return {};
    }

private:
    std::vector<Node> nodes_;
    Point start_, goal_;
    const cv::Mat& map_;
    std::mt19937 gen_;
    std::uniform_real_distribution<> x_dist_, y_dist_;
};

int main() {
    // 1. Setup Map
    cv::Mat map = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
    cv::rectangle(map, cv::Rect(200, 0, 50, 400), cv::Scalar(255), -1);
    cv::rectangle(map, cv::Rect(500, 200, 50, 400), cv::Scalar(255), -1);

    Point start = {50, 300};
    Point goal = {700, 300};

    // Visualization
    cv::Mat display;
    cv::cvtColor(map, display, cv::COLOR_GRAY2BGR);
    display = ~display; 
    
    cv::circle(display, cv::Point(start.x, start.y), 5, cv::Scalar(255, 0, 0), -1);
    cv::circle(display, cv::Point(goal.x, goal.y), 5, cv::Scalar(0, 0, 255), -1);

    RRT rrt(start, goal, map);

    fmt::print("Starting RRT...\n");
    bool reached = false;
    for (int i = 0; i < MAX_ITER; ++i) {
        if (rrt.step(display)) {
            reached = true;
            break;
        }
        
        if (i % 50 == 0) {
            cv::imshow("RRT", display);
            cv::waitKey(1);
        }
    }

    cv::waitKey(0);
    return 0;
}
