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
        int rows = grid.size();
        int cols = grid[0].size();

        auto cmp = [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            return a->f() > b->f();
        };
        std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, decltype(cmp)> open_set(cmp);
        std::unordered_map<Point, double, PointHash> g_scores;

        auto start_node = std::make_shared<Node>();
        start_node->pos = start;
        start_node->g = 0;
        start_node->h = heuristic(start, goal);
        
        open_set.push(start_node);
        g_scores[start] = 0;

        int visited_count = 0;

        while (!open_set.empty()) {
            auto current = open_set.top();
            open_set.pop();
            visited_count++;

            if (current->pos == goal) {
                return { reconstruct_path(current), visited_count };
            }

            if (current->g > g_scores[current->pos]) continue;

            const int dx[] = {0, 0, 1, -1};
            const int dy[] = {1, -1, 0, 0};

            for (int i = 0; i < 4; ++i) {
                Point next_pos = {current->pos.x + dx[i], current->pos.y + dy[i]};

                if (next_pos.x < 0 || next_pos.x >= cols || next_pos.y < 0 || next_pos.y >= rows) continue;
                if (grid[next_pos.y][next_pos.x] == 1) continue;

                double new_g = current->g + 1.0;

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

        return { {}, visited_count };
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
        return std::abs(a.x - b.x) + std::abs(a.y - b.y);
    });

    // 2. Run Dijkstra (Heuristic = 0)
    auto result_dijkstra = PathPlanner::search(grid, start, goal, [](Point a, Point b) {
        return 0.0;
    });

    fmt::print("Comparison Results:\n");
    fmt::print("--------------------------------------------------\n");
    fmt::print("Algorithm | Path Length | Nodes Visited | Efficiency\n");
    fmt::print("--------------------------------------------------\n");
    fmt::print("A*        | {:<11} | {:<13} | High\n", result_astar.path.size(), result_astar.nodes_visited);
    fmt::print("Dijkstra  | {:<11} | {:<13} | Low\n", result_dijkstra.path.size(), result_dijkstra.nodes_visited);
    fmt::print("--------------------------------------------------\n");

    if (result_astar.path.size() == result_dijkstra.path.size()) {
        fmt::print("[PASS] Both algorithms found optimal path.\n");
    } else {
        fmt::print("[FAIL] Path lengths differ!\n");
    }

    if (result_dijkstra.nodes_visited >= result_astar.nodes_visited) {
        fmt::print("[PASS] Dijkstra visited >= nodes than A*.\n");
    } else {
        fmt::print("[FAIL] Dijkstra visited FEWER nodes? Impossible if A* heuristic is consistent.\n");
    }

    return 0;
}
