#include "astar.hpp"
#include <fmt/core.h>

void print_grid(const Grid& grid, const std::vector<Point>& path) {
    auto temp_grid = grid;
    // Mark path with 2
    for (const auto& p : path) {
        if (temp_grid[p.y][p.x] == 0) temp_grid[p.y][p.x] = 2;
    }
    // Mark start/end
    if (!path.empty()) {
        temp_grid[path.front().y][path.front().x] = 3; // S
        temp_grid[path.back().y][path.back().x] = 4;   // E
    }

    for (const auto& row : temp_grid) {
        for (int cell : row) {
            char c = '.';
            if (cell == 1) c = '#';
            else if (cell == 2) c = '*';
            else if (cell == 3) c = 'S';
            else if (cell == 4) c = 'E';
            fmt::print("{} ", c);
        }
        fmt::print("\n");
    }
}

int main() {
    // 0: Free, 1: Obstacle
    Grid grid = {
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 0, 0, 0}
    };

    Point start = {0, 0};
    Point goal = {4, 4};

    fmt::print("Searching path from ({},{}) to ({},{})\n", start.x, start.y, goal.x, goal.y);
    auto path = AStar::search(grid, start, goal);

    if (path.empty()) {
        fmt::print("No path found!\n");
    } else {
        fmt::print("Path found! Length: {}\n", path.size());
        print_grid(grid, path);
    }

    return 0;
}
