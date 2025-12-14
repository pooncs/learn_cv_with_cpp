# Exercise 05: Eigen Decomposition

## Goal
Compute principal axes of a 2D point cloud using Eigenvalues and Eigenvectors.

## Learning Objectives
1.  Understand the geometric interpretation of Eigenvectors (Principal Axes).
2.  Compute the Covariance Matrix of a set of points.
3.  Use `Eigen::SelfAdjointEigenSolver` for symmetric matrices.

## Practical Motivation
**Principal Component Analysis (PCA)** is fundamental in CV:
- **Orientation Estimation**: Finding the orientation of an object.
- **Dimensionality Reduction**: Reducing features from high-dim to low-dim.
- **Normal Estimation**: Estimating surface normals in 3D point clouds.

## Theory & Background

### Covariance Matrix
Given a set of centered points $P$ (columns are points), the covariance matrix is:
$$ \Sigma = \frac{1}{N-1} P P^T $$
The eigenvectors of $\Sigma$ represent the directions of maximum variance. The eigenvalues represent the magnitude of variance in those directions.

## Implementation Tasks

### Task 1: Compute Mean and Center Data
Calculate the centroid of the point cloud and subtract it from all points.

### Task 2: Compute Covariance Matrix
Compute $\Sigma = \frac{1}{N-1} \sum (p_i - \mu)(p_i - \mu)^T$.

### Task 3: Eigen Decomposition
Use `Eigen::SelfAdjointEigenSolver` to find eigenvalues and eigenvectors of $\Sigma$.

## Common Pitfalls
- Forgetting to center the data before computing $PP^T$.
- Eigen sorts eigenvalues in **increasing** order (smallest first). Usually, we want the largest (last).
