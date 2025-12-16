# Module 05 - Exercise 04: Normal Estimation

## Goal
Compute surface normals for a point cloud using Principal Component Analysis (PCA) on local neighborhoods.

## Concept: PCA for Normals
The surface normal at a point represents the direction perpendicular to the surface.
We can estimate it by looking at the local neighborhood of the point (e.g., its $k$ nearest neighbors).

## Analogy: The Flat Plate
*   **The Problem:** You have a cloud of dots, but you want to know which way the surface is facing at a specific dot.
*   **The Neighbors:** Look at the 10 closest dots around your target dot.
*   **The Plate (PCA):** Imagine trying to balance a flat dinner plate on top of those 10 dots. You wiggle it until it sits as flat as possible.
*   **The Normal:** Imagine a stick glued to the center of the plate, pointing straight up. That stick is the "Normal". It tells you "This part of the surface is facing Up/Left/Right".
*   **Math:** The "wiggling" is finding the direction where the dots are *flattest* (least variance). This is the eigenvector with the smallest eigenvalue.

## Theory & Background
1.  **Neighborhood**: Collect $k$ points around the query point $P$.
2.  **Covariance Matrix**: Compute the covariance matrix of these points.
    $$ C = \frac{1}{k} \sum_{i=1}^{k} (p_i - \bar{p})(p_i - \bar{p})^T $$
    Where $\bar{p}$ is the centroid of the neighbors.
3.  **Eigen Decomposition**: Solve $C v = \lambda v$.
    - The eigenvectors represent the principal axes of the distribution.
    - The eigenvector corresponding to the **smallest eigenvalue** points in the direction of least variance, which is the **surface normal**.

## Task
1.  Generate or load a simple point cloud (e.g., a plane or sphere).
2.  Implement a simple `findKNearestNeighbors` function (Brute force is fine for small data).
3.  Implement `computeNormal` using Eigen to solve the PCA.
4.  Compute normals for all points.

## Instructions
1.  Navigate to `todo/` directory.
2.  Open `src/main.cpp`.
3.  Implement KNN and Normal Estimation.
4.  Build and run.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build .
```
