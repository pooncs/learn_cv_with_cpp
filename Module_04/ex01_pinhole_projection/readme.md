# Exercise 01: Pinhole Projection

## Goal
Implement the Pinhole Camera Model projection function.

## Learning Objectives
1.  Understand the camera intrinsic matrix $K$.
2.  Perform perspective division ($x/z, y/z$).
3.  Map 3D points to 2D pixel coordinates.

## Theory & Background

### Pinhole Model
The projection of a 3D point $P_c = [X_c, Y_c, Z_c]^T$ (in camera frame) to pixel coordinates $p = [u, v]^T$ is:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

Where:
- $f_x, f_y$: Focal lengths (in pixels).
- $c_x, c_y$: Principal point (optical center).
- $s = Z_c$: Scale factor (depth).

The normalized coordinates are $x' = X_c/Z_c$ and $y' = Y_c/Z_c$.
Then:
$$ u = f_x \cdot x' + c_x $$
$$ v = f_y \cdot y' + c_y $$

## Implementation Tasks

### Task 1: Projection Function
Implement `project_points` that takes a `cv::Mat` of 3D points ($N \times 3$) and an intrinsics matrix $K$, and returns 2D points ($N \times 2$).

## Common Pitfalls
- Division by zero if $Z_c = 0$.
- Points behind the camera ($Z_c < 0$).
