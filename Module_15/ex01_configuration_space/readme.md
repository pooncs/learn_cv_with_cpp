# Exercise 01: Configuration Space (C-Space)

## Goal
Visualize the Configuration Space (C-Space) for a simple circular robot in a 2D environment with rectangular obstacles.

## Learning Objectives
1.  **Understand C-Space**: Learn how a robot with geometry (radius) can be reduced to a single point by inflating obstacles.
2.  **Minkowski Sum**: Understand that C-Space obstacles are essentially the Minkowski sum of the workspace obstacle and the robot geometry.
3.  **Collision Checking Simplification**: Realize that collision checking becomes a simple point-in-polygon test in C-Space.

## Practical Motivation
Path planning algorithms (like A* or RRT) often treat the robot as a single point to simplify calculations. To make this valid, we must transform the world map into C-Space.
**Analogy:** Imagine you are walking through a doorway holding a wide horizontal pole. To you (the planning "point" in your head), the doorway feels much narrower. If the pole is 1 meter wide, the doorway is effectively 1 meter narrower. The "walls" have "inflated" inwards by half the pole's width on each side.

## Step-by-Step Instructions
1.  **Define Workspace**: Create a blank 2D map (image).
2.  **Add Obstacles**: Draw rectangular obstacles.
3.  **Define Robot**: Assume a circular robot with radius `r`.
4.  **Compute C-Space Obstacles**: Inflate each rectangular obstacle by `r` in all directions.
    *   For a rectangle, this results in a larger rounded rectangle (corners become quarter-circles). For simplicity in this exercise, we can approximate by expanding the rectangle dimensions by `r` on all sides (conservative approximation) or computing the exact rounded shape. Let's do the exact shape for visual clarity.
5.  **Visualize**: Show the Workspace (actual obstacles) and C-Space (inflated obstacles) side-by-side.
6.  **Interactive Check**: (Optional) Allow clicking to check if a configuration (point) is valid.

## Todo
1.  Implement `drawObstacle` to draw the original obstacle.
2.  Implement `drawCSpaceObstacle` to draw the inflated obstacle.
3.  Visualize the result.

## Verification
*   The C-Space image should show "fatter" obstacles than the Workspace image.
*   A point on the edge of a C-Space obstacle corresponds to the robot just touching the workspace obstacle.
