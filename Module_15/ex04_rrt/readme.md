# Exercise 04: Rapidly-exploring Random Trees (RRT)

## Goal
Implement RRT to find a path in a continuous 2D space (not a grid).

## Learning Objectives
1.  **Sampling-based Planning:** Unlike A* (grid-based), RRT works by randomly sampling the continuous space.
2.  **Tree Expansion:** `Nearest`, `Steer`, and `Extend` operations.
3.  **Probabilistic Completeness:** Understand that RRT finds a solution if one exists, given infinite time.

## Practical Motivation
For high-dimensional spaces (e.g., a 7-DOF robot arm), grids are impossible (curse of dimensionality). RRT is efficient in these high-dimensional spaces because it doesn't need to discretize the entire world.
**Analogy:** Imagine a tree growing roots. The roots branch out randomly looking for water (the goal). When a root hits a rock (obstacle), it stops, but other roots keep growing around it. Eventually, one root tip touches the water. The path of nutrients flows from the trunk to that specific tip.

## Step-by-Step Instructions
1.  **Sample:** Pick a random point $q_{rand}$ in space.
2.  **Nearest:** Find the closest node $q_{near}$ in the existing tree.
3.  **Steer:** Move from $q_{near}$ towards $q_{rand}$ by a fixed step size $\Delta$, creating $q_{new}$.
4.  **Collision Check:** If edge $(q_{near}, q_{new})$ is collision-free, add $q_{new}$ to tree.
5.  **Goal Check:** If $q_{new}$ is within a threshold distance to goal, add goal to tree and terminate.

## Todo
1.  Implement `getRandomPoint()`: Generate random (x, y).
2.  Implement `getNearestNode()`: Iterate through all nodes to find the closest one (Euclidean distance).
3.  Implement `steer()`: Calculate vector direction and move `step_size`.
4.  Implement `checkCollision()`: Check if the line segment intersects any obstacle (simple line-circle or line-rect intersection).
5.  Visualize the tree growing in real-time using `cv::imshow`.

## Verification
*   The visualization should show a tree branching out from the start like lightning.
*   The path will look "jagged" and unoptimized (RRT is not optimal).
