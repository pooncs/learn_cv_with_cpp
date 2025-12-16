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
        
        // Random number generator
        std::random_device rd;
        gen_ = std::mt19937(rd());
        x_dist_ = std::uniform_real_distribution<>(0, WIDTH);
        y_dist_ = std::uniform_real_distribution<>(0, HEIGHT);
    }

    // Step 1: Sample
    Point getRandomPoint() {
        // Bias towards goal occasionally (e.g., 10% chance)
        if (std::uniform_real_distribution<>(0, 1)(gen_) < 0.1) {
            return goal_;
        }
        return { (float)x_dist_(gen_), (float)y_dist_(gen_) };
    }

    // Step 2: Nearest
    int getNearestNodeIdx(Point query) {
        int nearest_idx = -1;
        float min_dist = std::numeric_limits<float>::max();

        for (int i = 0; i < nodes_.size(); ++i) {
            float d = dist(nodes_[i].pos, query);
            if (d < min_dist) {
                min_dist = d;
                nearest_idx = i;
            }
        }
        return nearest_idx;
    }

    // Step 3: Steer
    Point steer(Point from, Point to) {
        float d = dist(from, to);
        if (d < STEP_SIZE) return to;

        float theta = std::atan2(to.y - from.y, to.x - from.x);
        return {
            from.x + STEP_SIZE * std::cos(theta),
            from.y + STEP_SIZE * std::sin(theta)
        };
    }

    // Step 4: Collision Check (Image based)
    bool isCollisionFree(Point from, Point to) {
        // Check samples along the line
        int steps = std::ceil(dist(from, to) / 2.0); // Check every 2 pixels
        for (int i = 0; i <= steps; ++i) {
            float t = (float)i / steps;
            int x = (int)(from.x + t * (to.x - from.x));
            int y = (int)(from.y + t * (to.y - from.y));
            
            if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return false;
            // Assuming map: 0 = Free, 255 = Obstacle (or > 0)
            if (map_.at<uchar>(y, x) > 0) return false; 
        }
        return true;
    }

    // Main Loop Step
    bool step(cv::Mat& debug_img) {
        Point rand_pt = getRandomPoint();
        int nearest_idx = getNearestNodeIdx(rand_pt);
        Point nearest_pt = nodes_[nearest_idx].pos;
        Point new_pt = steer(nearest_pt, rand_pt);

        if (isCollisionFree(nearest_pt, new_pt)) {
            nodes_.push_back({new_pt, nearest_idx});
            
            // Visualization
            cv::line(debug_img, cv::Point(nearest_pt.x, nearest_pt.y), 
                     cv::Point(new_pt.x, new_pt.y), cv::Scalar(0, 255, 0), 1);
            cv::circle(debug_img, cv::Point(new_pt.x, new_pt.y), 2, cv::Scalar(0, 0, 255), -1);

            // Check Goal
            if (dist(new_pt, goal_) < GOAL_THRESHOLD) {
                nodes_.push_back({goal_, (int)nodes_.size() - 1});
                fmt::print("Goal Reached!\n");
                return true; // Reached
            }
        }
        return false;
    }

    std::vector<Point> getPath() {
        if (nodes_.back().pos.x != goal_.x || nodes_.back().pos.y != goal_.y) return {};
        
        std::vector<Point> path;
        int curr = nodes_.size() - 1;
        while (curr != -1) {
            path.push_back(nodes_[curr].pos);
            curr = nodes_[curr].parent_idx;
        }
        std::reverse(path.begin(), path.end());
        return path;
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
    // Draw obstacles (White = obstacle)
    cv::rectangle(map, cv::Rect(200, 0, 50, 400), cv::Scalar(255), -1);
    cv::rectangle(map, cv::Rect(500, 200, 50, 400), cv::Scalar(255), -1);

    Point start = {50, 300};
    Point goal = {700, 300};

    // Visualization
    cv::Mat display;
    cv::cvtColor(map, display, cv::COLOR_GRAY2BGR);
    // Invert map for visual: Black obstacles, White free space? 
    // Usually map is 0=free, 255=obs.
    // Let's invert display so obstacles are black, space is white.
    display = ~display; 
    
    // Draw Start/Goal
    cv::circle(display, cv::Point(start.x, start.y), 5, cv::Scalar(255, 0, 0), -1); // Blue Start
    cv::circle(display, cv::Point(goal.x, goal.y), 5, cv::Scalar(0, 0, 255), -1);   // Red Goal

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

    if (reached) {
        auto path = rrt.getPath();
        for (size_t i = 0; i < path.size() - 1; ++i) {
            cv::line(display, cv::Point(path[i].x, path[i].y), 
                     cv::Point(path[i+1].x, path[i+1].y), cv::Scalar(255, 0, 0), 2);
        }
        fmt::print("Path found with {} nodes.\n", path.size());
    } else {
        fmt::print("Failed to find path.\n");
    }

    cv::imshow("RRT", display);
    cv::waitKey(0);

    return 0;
}
