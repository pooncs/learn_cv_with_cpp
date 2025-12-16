# Exercise 07: Quaternions and SLERP

## Goal
Implement robust rotation interpolation (SLERP) using Eigen Quaternions.

## Learning Objectives
1.  **Quaternion Basics:** Understanding the 4D number system ($w, x, y, z$) used for 3D rotation.
2.  **SLERP (Spherical Linear Interpolation):** Smoothly rotating from A to B.
3.  **Why Quaternions?** No Gimbal Lock, compact, easy to interpolate.

## Analogy: The Shortest Flight Path
*   **The Problem:** You want to rotate a camera from "Looking Left" to "Looking Right".
*   **Linear Interpolation (Lerp) on Matrices:** Like flying from London to Tokyo by **digging a tunnel through the Earth**.
    *   The path is straight, but the "speed" (angular velocity) varies wildly. It distorts the rotation midway.
*   **SLERP (on Quaternions):** Like flying along the **surface of the Earth** (Great Circle).
    *   The path is the shortest arc.
    *   The speed is constant.
    *   The transition is perfectly smooth.

## Practical Motivation
When animating a camera or robot end-effector, you need smooth motion.
*   **Interpolation:** "At time $t=0$, be at orientation A. At $t=1$, be at B."
*   If you use Euler angles, the camera might swing wildly (Gimbal Lock).
*   If you use Matrices, you have to orthonormalize them constantly.
*   **Quaternions** are the standard for interpolation in Computer Graphics and Robotics (ROS uses them everywhere).

## Step-by-Step Instructions

### Task 1: Create Quaternions
Open `src/main.cpp`.
*   Create two rotations:
    *   $Q_1$: Identity (No rotation).
    *   $Q_2$: Rotation of 90 degrees around the Z-axis.
*   Use `Eigen::Quaterniond(Eigen::AngleAxisd(...))`.

### Task 2: Interpolate (SLERP)
*   Interpolate between $Q_1$ and $Q_2$ at $t = 0.5$ (Halfway).
*   Use `q1.slerp(0.5, q2)`.
*   Convert the result to Euler angles or Axis-Angle to verify it is indeed 45 degrees around Z.

### Task 3: The "Long Way" Around (Optional)
*   Try interpolating between $Q_1$ and $-Q_2$.
*   Note: $Q$ and $-Q$ represent the **same orientation** but are on opposite sides of the 4D sphere.
*   Interpolating to $-Q_2$ takes the "long way" (270 degrees) instead of the short way (90 degrees).
*   *Eigen's slerp handles this check automatically, but it's good to know.*

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show the interpolated angle being exactly halfway.
