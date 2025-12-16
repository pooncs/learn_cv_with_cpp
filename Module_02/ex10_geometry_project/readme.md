# Exercise 10: Geometry Project - 3D Coordinate Transformer

## Goal
Implement a full 3D coordinate transformer class for camera-to-world projection.

## Learning Objectives
1.  **The Full Pipeline:** World -> Camera -> Image (and back).
2.  **Intrinsics ($K$):** Focal length and center.
3.  **Extrinsics ($T$):** Rotation and Translation.

## Analogy: The Artist and the Window
*   **The World:** The 3D scene outside (Trees, Cars).
*   **The Camera ($T_{cw}$):** Where the Artist is standing relative to the scene.
    *   Moving the artist changes the view.
*   **The Canvas ($K$):** The window grid the Artist draws on.
    *   Small grid cells vs big grid cells (Focal Length).
    *   Where the center of the grid is (Principal Point).
*   **Projection:** Drawing the tree on the canvas.
*   **Back-Projection:** Seeing a dot on the canvas and calculating where the real tree is.
    *   *Critical:* You need to know **how far away** (Depth) the tree is, otherwise it could be a small bush close by or a huge redwood far away.

## Practical Motivation
This is the core logic of any SLAM or AR system.
*   **AR:** "Draw a Pikachu on the floor". You need to know where the floor is in the image (Project).
*   **SLAM:** "I see a corner at pixel (100, 200). Where is it in the map?" (Back-Project).

## Theory: The Pinhole Model
$$ p_{pixel} = \pi( K \times T_{cw} \times p_{world} ) $$
1.  **World to Camera:** $p_{cam} = T_{cw} \times p_{world}$ (Rigid Body Transform).
2.  **Projection:** Divide by Z. $p_{norm} = [x_c/z_c, y_c/z_c, 1]^T$.
3.  **Intrinsics:** Scale and Shift. $p_{pixel} = K \times p_{norm}$.
    $$ K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} $$

## Step-by-Step Instructions

### Task 1: Transformer Class
Open `src/transformer.cpp` (and header).
*   Member: `Eigen::Matrix3d K` (Intrinsics).
*   Member: `Eigen::Matrix4d T_cw` (World-to-Camera Extrinsics).
*   Constructor: Initialize these.

### Task 2: Project (World -> Image)
*   Implement `Eigen::Vector2d project(const Eigen::Vector3d& p_world)`.
    1.  Transform to Camera Frame: $p_{cam} = R p_{world} + t$ (or using $4 \times 4$ matrix).
    2.  Check if $z_{cam} <= 0$ (Behind camera). Handle error (return -1, -1 or throw).
    3.  Normalize: $x' = x/z, y' = y/z$.
    4.  Apply K: $u = f_x x' + c_x, v = f_y y' + c_y$.

### Task 3: Back Project (Image + Depth -> World)
*   Implement `Eigen::Vector3d back_project(const Eigen::Vector2d& p_pixel, double depth)`.
    1.  Undo K: $x' = (u - c_x)/f_x, y' = (v - c_y)/f_y$.
    2.  Scale by Depth: $p_{cam} = [x' \cdot Z, y' \cdot Z, Z]^T$.
    3.  Transform to World: $p_{world} = T_{cw}^{-1} \times p_{cam}$.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
The test should verify that `project(back_project(p))` returns the original pixel.
