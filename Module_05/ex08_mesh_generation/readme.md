# Module 05 - Exercise 08: Mesh Generation

## Goal
Create a surface mesh (triangles) from a set of 3D points.

## Concept: Meshing
A point cloud is just a set of unconnected samples. A **mesh** connects these points with triangles to form a continuous surface.

## Analogy: The Fishing Net
*   **Point Cloud:** Imagine a swarm of fireflies flying in the shape of a dragon. If you get close, you see they are just separate dots.
*   **Mesh:** Imagine throwing a net over the fireflies.
*   **Vertices:** The knots in the net correspond to the fireflies (Points).
*   **Faces (Triangles):** The threads connecting the knots form a solid surface (Skin).
*   **Result:** Now you have a solid dragon, not just a cloud of dots.

### Structured vs Unstructured
1.  **Structured (e.g., Depth Map)**: We know the neighbor relationship (pixels $(u,v)$ are neighbors). We can simply triangulate adjacent pixels.
2.  **Unstructured (e.g., Lidar)**: We don't know who is next to whom. Algorithms like **Poisson Surface Reconstruction** or **Ball Pivoting** are used.

## Task
We will implement a **Structured Mesh Generator** (like a simplified version of what happens when rendering a depth map).
1.  Generate a grid of points (representing a depth map).
2.  For each "pixel" $(i, j)$, create two triangles connecting it to $(i+1, j)$, $(i, j+1)$, and $(i+1, j+1)$.
3.  Save the result as an **OBJ** or **PLY** file (which supports faces).

## Instructions
1.  Navigate to `todo/` directory.
2.  Open `src/main.cpp`.
3.  Implement the triangulation logic.
4.  Build and run.

## Build
```bash
mkdir build
cd build
cmake ..
cmake --build .
```
