#pragma once
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>

struct Point {
    int x, y;
    bool operator==(const Point& other) const { return x == other.x && y == other.y; }
    bool operator!=(const Point& other) const { return !(*this == other); }
};

struct Node {
    Point pos;
    double g = 0.0;
    double h = 0.0;
    std::shared_ptr<Node> parent = nullptr;

    double f() const { return g + h; }
};

using Grid = std::vector<std::vector<int>>; // 0 = free, 1 = obstacle

class AStar {
public:
    static std::vector<Point> search(const Grid& grid, Point start, Point goal) {
        // TODO: Implement A* search
        // 1. Initialize open_set (priority_queue) and g_scores map
        // 2. Add start node to open_set
        // 3. While open_set not empty:
        //    a. Pop node with lowest f
        //    b. If goal, reconstruct path
        //    c. For each neighbor:
        //       i. Calculate new_g
        //       ii. If better path, update and push to open_set
        return {}; 
    }
};
