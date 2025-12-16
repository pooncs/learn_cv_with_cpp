# Exercise 01: Pinhole Projection

## Goal
Implement the Pinhole Camera Model projection function.

## Learning Objectives
1.  Understand the camera intrinsic matrix $K$.
2.  Perform perspective division ($x/z, y/z$).
3.  Map 3D points to 2D pixel coordinates.

## Analogy: The Artist's Window
Imagine you are an artist trying to paint a landscape on a glass window.
*   **The World ($X, Y, Z$):** The trees and mountains outside.
*   **The Eye (Camera Center):** Where you are standing.
*   **The Glass (Image Plane):** The surface you paint on.
*   **Focal Length ($f$):** The distance between your eye and the glass.
    *   *Glass close to eye (Small $f$):* You see a wide view (Wide Angle).
    *   *Glass far from eye (Large $f$):* You see a narrow view (Telephoto).
*   **Principal Point ($c_x, c_y$):** The exact point on the glass directly in front of your eye.
*   **Projection:** To paint a tree top, you draw a straight line from your eye to the tree top. Where that line hits the glass is where you paint the dot.

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
