# Exercise 06: ICP Alignment

## Goal
Use Open3D (or a manual implementation) to align two point clouds using Iterative Closest Point.

## Learning Objectives
1.  Understand the ICP algorithm (Associate -> Estimate Transform -> Apply -> Repeat).
2.  Align a source cloud to a target cloud.
3.  Evaluate alignment quality (RMSE).

## Analogy: The Magnetic Slide
*   **The Problem:** You have two identical plastic shapes, but one is rotated and shifted. You want to lock them together.
*   **ICP (Iterative Closest Point):**
    1.  **Magnetize:** Imagine every point on Shape A is a magnet attracted to the *closest* point on Shape B.
    2.  **Slide:** The magnets pull Shape A slightly closer to Shape B.
    3.  **Re-evaluate:** Now that A has moved, the "closest points" might have changed. So we check the magnets again.
    4.  **Repeat:** We keep sliding and re-checking until the shapes "snap" together perfectly.

## Theory & Background

### ICP
Given Source $P$ and Target $Q$.
1.  For each $p_i \in P$, find closest $q_j \in Q$.
2.  Find $R, t$ that minimizes $\sum || R p_i + t - q_j ||^2$.
3.  $P \leftarrow R P + t$.
4.  Repeat until convergence.

## Implementation Tasks

### Task 1: Manual ICP (Simplified)
Implement a basic Point-to-Point ICP loop (using brute force NN).

### Task 2: Transformation Estimation
Use SVD to find $R, t$ (Kabsch algorithm).

## Common Pitfalls
- Local minima (requires good initial guess).
- Outliers affecting the least squares solution.
