# Module 05 - Exercise 07: Colored Point Clouds

## Goal
Map RGB texture from an image onto a 3D point cloud.

## Concept: Texture Mapping
To color a 3D point $P=(X, Y, Z)$, we project it back onto the 2D image plane using the camera intrinsics.
$$
u = \frac{f_x \cdot X}{Z} + c_x \\
v = \frac{f_y \cdot Y}{Z} + c_y
$$
We then sample the color $C(u, v)$ from the RGB image and assign it to the point.

## Analogy: The Slide Projector
*   **The Object:** A plain white statue (The uncolored Point Cloud).
*   **The Projector:** The Camera (The RGB Image).
*   **The Process:** Imagine shining the colorful photo onto the white statue using a projector.
*   **Texture Mapping:** If a red pixel from the photo hits the nose of the statue, we paint the nose red.
*   **Math:** We calculate exactly which pixel "shines" on which 3D point.

## Task
1.  Create a synthetic point cloud (e.g., a plane).
2.  Create a synthetic RGB image (e.g., a checkerboard pattern).
3.  Implement `mapColorToCloud` which projects each point to $(u, v)$ and fetches color.
4.  Save the result to a PLY file (PLY supports color properties).

## Instructions
1.  Navigate to `todo/` directory.
2.  Open `src/main.cpp`.
3.  Implement the projection and color lookup.
4.  Build and run.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build .
```
