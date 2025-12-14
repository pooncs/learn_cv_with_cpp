# Exercise 05: Homography

## Goal
Compute and apply a Homography matrix to warp an image.

## Learning Objectives
1.  Understand the 8-DOF Homography transformation.
2.  Use `cv::findHomography` with 4 point correspondences.
3.  Use `cv::warpPerspective` to apply the transformation.

## Theory & Background

### Homography
A homography $H$ maps points from one plane to another.
$$ s \begin{bmatrix} u' \\ v' \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} $$

It preserves lines but not parallelism (unlike Affine).

## Implementation Tasks

### Task 1: Compute H
Given 4 source points and 4 destination points, compute $H$.

### Task 2: Warp Image
Apply $H$ to an image.

## Common Pitfalls
- Points must be in the same order (e.g., clockwise starting from top-left).
- Collinear points result in a degenerate solution.
