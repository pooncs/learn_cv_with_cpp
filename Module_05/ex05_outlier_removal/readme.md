# Module 05 - Exercise 05: Statistical Outlier Removal

## Goal
Implement a Statistical Outlier Removal (SOR) filter to clean up noisy point clouds.

## Concept: SOR Filter
Real-world sensors (like Lidar or RGB-D cameras) often produce "ghost" points or measurement noise.
The SOR filter removes points that are further away from their neighbors compared to the average point.

## Analogy: The Loner Detection
*   **The Crowd:** A group of people (points) standing together.
*   **The Loner:** Someone standing far away from everyone else (Noise).
*   **The Rule (SOR):**
    1.  Measure how far everyone is from their 5 closest friends.
    2.  Calculate the "Average Friend Distance" for the whole group.
    3.  If someone's friend distance is WAY bigger than the average (e.g., 2 times bigger), they are a Loner.
    4.  **Action:** Remove the Loners to clean up the crowd.

## Theory & Background
**Algorithm:**
1.  **KNN Mean Distance**: For every point $p_i$, calculate the mean distance $\bar{d}_i$ to its $k$ nearest neighbors.
2.  **Global Statistics**: Calculate the mean $\mu$ and standard deviation $\sigma$ of the distribution of all $\bar{d}_i$.
3.  **Threshold**: Define a threshold $T = \mu + \alpha \cdot \sigma$.
    - $\alpha$ is a multiplier (typically 1.0 to 3.0).
4.  **Filter**: Keep only points where $\bar{d}_i < T$.

## Task
1.  Create a point cloud with some intentional outliers (points far from the main cluster).
2.  Implement `computeMeanDistances` using KNN.
3.  Compute global $\mu$ and $\sigma$.
4.  Filter the cloud and return only inliers.

## Instructions
1.  Navigate to `todo/` directory.
2.  Open `src/main.cpp`.
3.  Implement the SOR algorithm.
4.  Build and run.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build .
```
