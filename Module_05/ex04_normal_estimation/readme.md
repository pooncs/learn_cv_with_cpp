# Exercise 04: Normal Estimation

## Goal
Compute surface normals using k-nearest neighbors and PCA.

## Learning Objectives
1.  Find k-nearest neighbors (Brute force for now, or simple grid).
2.  Compute Covariance Matrix of neighbors.
3.  Perform Eigen decomposition to find the normal (eigenvector with smallest eigenvalue).

## Theory & Background

### PCA Normal Estimation
For a point $P$ and its neighbors $N_k(P)$:
1.  Compute centroid $\bar{P}$.
2.  Compute Covariance Matrix $C = \sum (P_i - \bar{P})(P_i - \bar{P})^T$.
3.  Solve $C v = \lambda v$.
4.  Normal $n$ is the eigenvector corresponding to $\lambda_{min}$.

## Implementation Tasks

### Task 1: Find Neighbors
Implement a simple `find_knn` (linear search $O(N)$ per query, total $O(N^2)$).

### Task 2: Compute Normal
Implement `compute_normal(neighbors)`.

## Common Pitfalls
- Normal orientation (ambiguity).
- Degenerate cases (collinear points).
