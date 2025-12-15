# Exercise 02: A* Search on a Grid Map

## Goal
Implement the A* (A-Star) search algorithm to find the shortest path between two points on a 2D grid map with obstacles.

## Learning Objectives
1.  **Pathfinding:** Understand how A* combines Dijkstra's algorithm (g-cost) with a heuristic (h-cost) to guide the search.
2.  **Heuristics:** Learn to use Manhattan distance vs. Euclidean distance as heuristics.
3.  **Priority Queues:** Use `std::priority_queue` to efficiently select the next node to explore.
4.  **Grid Maps:** Represent a 2D world as a vector of vectors or a flat array.

## Practical Motivation
A* is the backbone of path planning in robotics and games. Whether it's a Roomba cleaning a floor or a character navigating a maze, A* ensures the robot gets from A to B efficiently without hitting walls.

## Theory: The A* Algorithm
A* minimizes the cost function: $f(n) = g(n) + h(n)$
*   $g(n)$: Cost from start to node $n$ (actual distance traveled).
*   $h(n)$: Estimated cost from $n$ to goal (heuristic).

**Algorithm Steps:**
1.  Open Set (Priority Queue): Contains nodes to be evaluated, sorted by $f(n)$.
2.  Closed Set: Nodes already evaluated.
3.  While Open Set is not empty:
    *   Pop node `current` with lowest $f$.
    *   If `current` is goal, reconstruct path.
    *   Add `current` to Closed Set.
    *   For each neighbor:
        *   If neighbor in Closed Set or is obstacle, skip.
        *   Calculate tentative $g_{new} = g(current) + dist(current, neighbor)$.
        *   If $g_{new} < g(neighbor)$ or neighbor not in Open Set:
            *   Update $g(neighbor) = g_{new}$
            *   Update $h(neighbor) = heuristic(neighbor, goal)$
            *   Update parent of neighbor to `current`.
            *   Add to Open Set.

## Step-by-Step Instructions

### Task 1: Define the Grid
Represent the map as a `std::vector<std::vector<int>>` where 0 is free space and 1 is an obstacle.

### Task 2: Implement the Node Structure
Create a struct `Node` holding coordinates `(x, y)`, `g_cost`, `h_cost`, and a pointer/index to the `parent` node for path reconstruction.

### Task 3: Implement A*
Fill in the `find_path` function in `src/astar.cpp`.
*   Use `std::priority_queue` with a custom comparator.
*   Implement the Manhattan heuristic: $|x1 - x2| + |y1 - y2|$.

### Task 4: Path Reconstruction
Once the goal is reached, backtrack from the goal node to the start node using the parent pointers to generate the path.

## Code Hints
*   **Priority Queue:** `std::priority_queue` pops the *largest* element by default. To make it a min-queue (pop smallest $f$), use `std::greater` or a custom struct.
    ```cpp
    struct CompareNode {
        bool operator()(const Node* a, const Node* b) {
            return a->f_cost() > b->f_cost();
        }
    };
    ```
*   **Neighbors:** For a grid, neighbors are (x+1, y), (x-1, y), (x, y+1), (x, y-1). Check bounds!

## Verification
Run the tests.
*   Test 1: Simple path in an empty grid.
*   Test 2: Path around a wall.
*   Test 3: No path possible (should return empty).
