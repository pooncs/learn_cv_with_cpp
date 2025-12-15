#include <gtest/gtest.h>
#include "astar.hpp"

TEST(AStarTest, SimplePath) {
    Grid grid = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };
    Point start = {0, 0};
    Point goal = {2, 2};
    auto path = AStar::search(grid, start, goal);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(path.front(), start);
    EXPECT_EQ(path.back(), goal);
}

TEST(AStarTest, ObstacleAvoidance) {
    Grid grid = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 1, 0}
    };
    Point start = {0, 0};
    Point goal = {0, 2};
    // Direct path blocked by (1,1) and (1,2)?? No, (0,0)->(0,1)->(0,2) is free.
    // Let's block the middle
    // 0 1 0
    // 0 1 0
    // 0 0 0
    // Start (0,0) Goal (2,0). Path must go down and around.
    
    Grid wall_grid = {
        {0, 1, 0},
        {0, 1, 0},
        {0, 0, 0}
    };
    auto path = AStar::search(wall_grid, {0,0}, {2,0});
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(path.front(), (Point{0,0}));
    EXPECT_EQ(path.back(), (Point{2,0}));
    // Path should be at least 6 steps: (0,0)->(0,1)->(0,2)->(1,2)->(2,2)->(2,1)->(2,0) is 6 steps? 
    // Wait, (0,0)->(0,1)->(0,2)->(1,2)->(2,2)->(2,1)->(2,0)
    // Actually heuristic is admissible, so it finds shortest path.
}

TEST(AStarTest, NoPath) {
    Grid grid = {
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0}
    };
    auto path = AStar::search(grid, {0,0}, {2,0});
    EXPECT_TRUE(path.empty());
}
