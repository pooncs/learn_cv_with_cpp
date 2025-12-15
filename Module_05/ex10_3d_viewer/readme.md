# Exercise 10: 3D Viewer

## Goal
Create a simple 3D viewer using a library like OpenGL (via GLFW) or Open3D.
*Since we are sticking to core libraries, we will create a minimal OpenGL-based viewer or just a command-line "renderer" that outputs an image.*

**For this exercise, we will use a simple software rasterizer concept to project points to an image, as full OpenGL setup might be complex in this headless environment.**

## Learning Objectives
1.  Understand 3D-to-2D projection pipeline (Model -> View -> Projection).
2.  Implement a simple "LookAt" camera.
3.  Draw points on an image canvas.

## Theory & Background

### MVP Matrix
$$ p_{clip} = P \cdot V \cdot M \cdot p_{local} $$
- **Model**: Local to World.
- **View**: World to Camera ($[R | t]$).
- **Projection**: Camera to Clip space (Perspective).

### Viewport Transform
Map clip space $[-1, 1]$ to screen coordinates $[0, W] \times [0, H]$.

## Implementation Tasks

### Task 1: Camera Class
Implement `Camera` with `lookAt(eye, center, up)`.

### Task 2: Rasterizer
Implement `render_point_cloud(cloud, camera, width, height)` returning `cv::Mat`.

## Common Pitfalls
- Matrix multiplication order.
- Z-buffering (painter's algorithm: sort by depth).
