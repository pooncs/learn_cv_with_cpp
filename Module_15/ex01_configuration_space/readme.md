# Exercise 01: Configuration Space Visualization

## Goal
Visualize the Configuration Space (C-Space) for a simple 2D robot.

## Learning Objectives
1.  **C-Space Concept:** Understand that a robot with non-zero size in Workspace transforms into a point in C-Space by inflating obstacles.
2.  **Minkowski Sum:** Learn how inflating obstacles by the robot's radius creates the C-Space obstacles.
3.  **Visualization:** Use OpenCV to draw the original map and the computed C-Space.

## Practical Motivation
Path planning algorithms like A* treat the robot as a single point. To make this valid, we must first "fatten" the walls so the point-robot doesn't clip them.

## Step-by-Step Instructions
1.  Create a 2D binary grid map (Workspace).
2.  Define a circular robot with radius $R$.
3.  Compute C-Space obstacles: For every obstacle pixel, mark all pixels within distance $R$ as obstacles.
    *   *Hint:* This is equivalent to morphological Dilation.
4.  Visualize both maps side-by-side.

## Code Hints
*   **OpenCV Dilation:** `cv::dilate` with a circular structuring element is the efficient way to compute Minkowski sums for circular robots.
    ```cpp
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*r+1, 2*r+1));
    cv::dilate(map, c_space, element);
    ```

## Verification
*   The obstacles in the C-Space map should be thicker than in the Workspace map.
*   A point-robot moving in C-Space should never visually overlap with original obstacles when mapped back to Workspace.
