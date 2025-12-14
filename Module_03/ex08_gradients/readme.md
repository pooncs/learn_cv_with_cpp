# Exercise 08: Gradients

## Goal
Compute Sobel X and Y derivatives and gradient magnitude/orientation.

## Learning Objectives
1.  Understand image derivatives as convolution with finite difference kernels.
2.  Compute Gradient Magnitude and Orientation.
3.  Visualize gradients.

## Practical Motivation
Gradients are the basis of edge detection (Canny), feature extraction (HOG, SIFT), and optical flow.

## Theory & Background

### Sobel Operators
$$ G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I $$
$$ G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I $$

### Magnitude & Angle
$$ M = \sqrt{G_x^2 + G_y^2} $$
$$ \theta = \arctan(G_y / G_x) $$

## Implementation Tasks

### Task 1: Sobel X and Y
Compute $G_x$ and $G_y$ using OpenCV's `Sobel` or `filter2D`.

### Task 2: Magnitude and Angle
Compute magnitude and orientation (in degrees).

## Common Pitfalls
- **Data Types**: Gradients can be negative. Use `CV_16S` or `CV_32F`. Do not use `CV_8U` for intermediate results.
- **Angle Range**: `atan2` returns $[-\pi, \pi]$. Map to $[0, 360)$ or $[0, 180)$.
