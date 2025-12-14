# Exercise 02: Inverse Projection (Back-Projection)

## Goal
Map a 2D pixel coordinate to a 3D ray in camera space.

## Learning Objectives
1.  Understand the inverse of the intrinsic matrix $K^{-1}$.
2.  Convert pixel coordinates to normalized device coordinates (NDC).
3.  Recover 3D position given depth $Z$.

## Theory & Background

### Inverse Projection
Given a pixel $p = [u, v]^T$, we want to find the direction vector (ray) in 3D space.
From the pinhole model:
$$ Z \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} $$
$$ \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = Z \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} $$

Let $p_{hom} = [u, v, 1]^T$.
The normalized ray is $d = K^{-1} p_{hom}$.
The 3D point at depth $Z$ is $P = Z \cdot d$.

## Implementation Tasks

### Task 1: Pixel to Ray
Implement `pixel_to_ray(u, v, K)` returning a `cv::Point3d` direction.

### Task 2: Reconstruct Point
Implement `reconstruct_point(u, v, Z, K)`.

## Common Pitfalls
- Assuming $K$ is always invertible (it is for valid cameras).
- Matrix multiplication order.
