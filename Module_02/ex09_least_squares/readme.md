# Exercise 09: Least Squares Fitting

## Goal
Fit a plane to noisy 3D points using Singular Value Decomposition (SVD).

## Learning Objectives
1.  Formulate a geometric fitting problem as a least squares minimization.
2.  Use `Eigen::JacobiSVD` or `Eigen::BDCSVD` to solve homogeneous systems.
3.  Understand the relationship between the smallest singular value and the solution.

## Practical Motivation
In 3D Vision, we often need to fit geometric primitives (lines, planes, cylinders) to point clouds obtained from LiDAR or Depth cameras. RANSAC often uses a least-squares solver in its inner loop.

## Theory & Background

### Plane Equation
A plane is defined by $ax + by + cz + d = 0$.
We want to find $n = [a, b, c, d]^T$ such that $\sum (n^T p_i)^2$ is minimized, subject to $||n|| = 1$ (or $a^2+b^2+c^2=1$).

### SVD Solution
1.  Compute the centroid $\bar{p}$.
2.  Subtract centroid from all points: $P' = P - \bar{p}$.
3.  Form the matrix $A = P'^T$.
4.  Compute SVD of $A$.
5.  The normal vector $[a, b, c]^T$ corresponds to the singular vector associated with the **smallest singular value** (last column of V).
6.  Solve for $d = -n \cdot \bar{p}$.

Alternatively, one can solve $Ax=0$ directly using SVD on the homogeneous points, but centering first is numerically more stable.

## Implementation Tasks

### Task 1: Fit Plane
Implement a function that takes $N$ 3D points and returns the plane coefficients $(a, b, c, d)$.

### Task 2: Verify
Generate points on a known plane, add noise, and verify the fitted coefficients.

## Common Pitfalls
- **Normalization**: The normal vector returned by SVD is unit length.
- **Sign Ambiguity**: The normal can point in either direction ($n$ or $-n$).
