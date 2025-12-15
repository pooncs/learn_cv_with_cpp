# Exercise 08: PnP Solver

## Goal
Solve Perspective-n-Point to find camera pose from 3D-2D matches.

## Learning Objectives
1.  Understand the PnP problem ($[R|t]$ that maps $P_w$ to $p_{img}$).
2.  Use `cv::solvePnP`.
3.  Project 3D axes onto the image to visualize pose.

## Practical Motivation
Augmented Reality (AR) uses PnP to place virtual objects on real-world markers (like chessboards or QR codes).

## Theory & Background

### PnP
Given:
- $N$ 3D points in world frame $P_w$.
- $N$ corresponding 2D points $p$.
- Intrinsics $K$.

Find $R, t$ such that:
$$ s \cdot p = K (R P_w + t) $$

## Implementation Tasks

### Task 1: Estimate Pose
Implement `estimate_pose(object_points, image_points, K, dist)` returning $rvec, tvec$.

### Task 2: Draw Axes
Project $(0,0,0), (L,0,0), (0,L,0), (0,0,L)$ and draw lines.

## Common Pitfalls
- Coordinate frames (OpenCV uses y-down, z-forward).
- `rvec` is a rotation vector (Rodrigues), not a matrix.
