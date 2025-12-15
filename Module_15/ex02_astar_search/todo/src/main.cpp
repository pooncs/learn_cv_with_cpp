#include "astar.hpp"
#include <iostream>

int main() {
    Grid grid = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };
    Point start = {0, 0};
    Point goal = {2, 2};

    auto path = AStar::search(grid, start, goal);
    
    if (path.empty()) {
        std::cout << "TODO: Implement A* to find the path!" << std::endl;
    } else {
        std::cout << "Path found! Length: " << path.size() << std::endl;
    }
    return 0;
}
