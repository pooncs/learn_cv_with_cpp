# Exercise 10: Geometry Project - 3D Coordinate Transformer

## Goal
Implement a full 3D coordinate transformer class for camera-to-world projection.

## Learning Objectives
1.  Integrate SE(3) transformations and Pinhole Camera Models.
2.  Implement `project` (3D World -> 2D Pixel) and `back_project` (2D Pixel + Depth -> 3D World).
3.  Design a reusable C++ class for geometric operations.

## Practical Motivation
This is the core logic of any SLAM or AR system. You need to map 3D points in the world to 2D pixels on the screen (Rendering/Projection) and map 2D features back to 3D space (Triangulation/Back-projection).

## Theory & Background

### Pipeline
$$ p_{pixel} = \pi( K \times T_{cw} \times p_{world} ) $$
1.  **World to Camera**: $p_{cam} = T_{cw} \times p_{world}$ (Rigid Body Transform).
2.  **Projection**: $p_{norm} = [x_c/z_c, y_c/z_c, 1]^T$.
3.  **Intrinsics**: $p_{pixel} = K \times p_{norm}$.
    $$ K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} $$

### Back Projection
Given pixel $(u, v)$ and depth $Z$:
$$ p_{cam} = Z \times K^{-1} \times [u, v, 1]^T $$
$$ p_{world} = T_{cw}^{-1} \times p_{cam} = T_{wc} \times p_{cam} $$

## Implementation Tasks

### Task 1: Transformer Class
Create a class `Transformer` that stores intrinsics $K$ and extrinsics $T_{cw}$.

### Task 2: Project
Implement `Eigen::Vector2d project(const Eigen::Vector3d& p_world)`.

### Task 3: Back Project
Implement `Eigen::Vector3d back_project(const Eigen::Vector2d& p_pixel, double depth)`.

## Common Pitfalls
- **Extrinsics Direction**: Is $T$ camera-to-world ($T_{wc}$) or world-to-camera ($T_{cw}$)? Usually we store $T_{cw}$ to project points easily. But we need $T_{wc}$ ($T_{cw}^{-1}$) to move camera or back-project.
- **Division by Zero**: When $Z=0$.
