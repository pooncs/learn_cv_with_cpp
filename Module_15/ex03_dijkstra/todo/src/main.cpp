#include <fmt/core.h>
#include <vector>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <memory>
#include <functional>

// --- Structures ---
struct Point {
    int x, y;
    bool operator==(const Point& other) const { return x == other.x && y == other.y; }
};

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

using Grid = std::vector<std::vector<int>>;

// --- Generic Search ---
struct SearchResult {
    std::vector<Point> path;
    int nodes_visited = 0;
};

class PathPlanner {
public:
    // Heuristic Function Type
    using HeuristicFunc = std::function<double(Point, Point)>;

    static SearchResult search(const Grid& grid, Point start, Point goal, HeuristicFunc heuristic) {
        if (grid.empty() || grid[0].empty()) return {};
        
        // TODO: Implement the search logic here.
        // 1. Setup Priority Queue and Open/Closed sets.
        // 2. Loop until queue is empty.
        // 3. For A*: h = heuristic(pos, goal).
        // 4. For Dijkstra: h = 0 (or just pass a lambda that returns 0).
        // 5. Count nodes visited (popped from queue).
        
        return { {}, 0 };
    }

private:
    static std::vector<Point> reconstruct_path(std::shared_ptr<Node> node) {
        std::vector<Point> path;
        while (node) {
            path.push_back(node->pos);
            node = node->parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }
};

int main() {
    Grid grid(20, std::vector<int>(20, 0));
    // Add a wall
    for (int y = 5; y < 15; ++y) grid[y][10] = 1;

    Point start = {2, 10};
    Point goal = {18, 10};

    // 1. Run A* (Manhattan)
    auto result_astar = PathPlanner::search(grid, start, goal, [](Point a, Point b) {
        // TODO: Return Manhattan distance
        return 0.0; 
    });

    // 2. Run Dijkstra (Heuristic = 0)
    auto result_dijkstra = PathPlanner::search(grid, start, goal, [](Point a, Point b) {
        return 0.0;
    });

    fmt::print("Comparison Results:\n");
    fmt::print("A* Visited: {}\n", result_astar.nodes_visited);
    fmt::print("Dijkstra Visited: {}\n", result_dijkstra.nodes_visited);

    return 0;
}
