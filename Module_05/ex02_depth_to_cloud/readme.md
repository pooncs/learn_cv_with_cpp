# Module 05 - Exercise 02: Depth to Point Cloud

## Goal
Convert a 2D depth image into a 3D point cloud using camera intrinsic parameters.

## Concept: Back-Projection
A depth image contains the distance $Z$ from the camera center for each pixel $(u, v)$.
To recover the 3D coordinate $(X, Y, Z)$, we use the **Pinhole Camera Model**.

## Analogy: The Pin Art Toy
Remember those toys with thousands of metal pins? You press your hand into the back, and the pins stick out the front to show the 3D shape.
*   **Depth Image:** The back of the toy. Each pin's position is a pixel. How far you pushed it is the "Depth" value.
*   **Point Cloud:** The 3D shape of your hand that appears on the front.
*   **Back Projection:** The math that says "Pin #50 is at row 10, column 5, and it is pushed out 3cm. Therefore, the 3D point is at (X, Y, Z)".

## Theory & Background
$$
\begin{align*}
X &= \frac{(u - c_x) \cdot Z}{f_x} \\
Y &= \frac{(v - c_y) \cdot Z}{f_y} \\
Z &= \text{depth}(u, v)
\end{align*}
$$

Where:
- $(c_x, c_y)$ is the principal point (optical center).
- $(f_x, f_y)$ is the focal length in pixels.

## Task
1.  Use OpenCV to create or load a synthetic depth image (e.g., a gradient or a shape).
2.  Define a `CameraIntrinsics` struct.
3.  Implement `depthToCloud` that iterates over pixels and computes 3D points.
4.  Save the result to `output.xyz` (reusing your code from Ex 01 or writing a simple dumper).

## Instructions
1.  Navigate to `todo/` directory.
2.  Open `src/main.cpp`.
3.  Implement the back-projection logic.
4.  Build and run.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build .
```
