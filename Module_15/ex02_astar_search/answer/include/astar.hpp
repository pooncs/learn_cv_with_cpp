#pragma once
#include <vector>
#include <cmath>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <memory>

struct Point {
    int x, y;
    bool operator==(const Point& other) const { return x == other.x && y == other.y; }
    bool operator!=(const Point& other) const { return !(*this == other); }
};

// Simple hash for Point to be used in unordered_map/set
struct PointHash {
    std::size_t operator()(const Point& p) const {
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
    }
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
        if (grid.empty() || grid[0].empty()) return {};
        int rows = grid.size();
        int cols = grid[0].size();

        // Min-priority queue
        auto cmp = [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            return a->f() > b->f();
        };
        std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, decltype(cmp)> open_set(cmp);

        // Keep track of visited nodes and lowest g-score found so far
        std::unordered_map<Point, double, PointHash> g_scores;

        auto start_node = std::make_shared<Node>();
        start_node->pos = start;
        start_node->g = 0;
        start_node->h = heuristic(start, goal);
        
        open_set.push(start_node);
        g_scores[start] = 0;

        while (!open_set.empty()) {
            auto current = open_set.top();
            open_set.pop();

            if (current->pos == goal) {
                return reconstruct_path(current);
            }

            // If we found a shorter path to this node already, skip
            if (current->g > g_scores[current->pos]) continue;

            // Neighbors (4-connectivity)
            const int dx[] = {0, 0, 1, -1};
            const int dy[] = {1, -1, 0, 0};

            for (int i = 0; i < 4; ++i) {
                Point next_pos = {current->pos.x + dx[i], current->pos.y + dy[i]};

                // Bounds check
                if (next_pos.x < 0 || next_pos.x >= cols || next_pos.y < 0 || next_pos.y >= rows) continue;
                // Obstacle check
                if (grid[next_pos.y][next_pos.x] == 1) continue;

                double new_g = current->g + 1.0; // Assume uniform cost

                if (g_scores.find(next_pos) == g_scores.end() || new_g < g_scores[next_pos]) {
                    g_scores[next_pos] = new_g;
                    auto next_node = std::make_shared<Node>();
                    next_node->pos = next_pos;
                    next_node->g = new_g;
                    next_node->h = heuristic(next_pos, goal);
                    next_node->parent = current;
                    open_set.push(next_node);
                }
            }
        }

        return {}; // No path found
    }

private:
    static double heuristic(Point a, Point b) {
        // Manhattan distance
        return std::abs(a.x - b.x) + std::abs(a.y - b.y);
    }

    static std::vector<Point> reconstruct_path(std::shared_ptr<Node> end_node) {
        std::vector<Point> path;
        auto current = end_node;
        while (current) {
            path.push_back(current->pos);
            current = current->parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }
};
