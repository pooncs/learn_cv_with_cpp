# Exercise 06: Rotation Representations

## Goal
Convert between Rotation Matrices (SO3), Euler Angles, and Axis-Angle representations.

## Learning Objectives
1.  Understand the 3 common rotation representations and their pros/cons.
2.  Use `Eigen::AngleAxis` and `Eigen::Matrix3d`.
3.  Implement conversion logic (especially Euler angles ZYX).

## Practical Motivation
- **Sensors**: IMUs often give Euler angles or Quaternions.
- **Control**: Robots are often controlled in joint angles (Euler-like) or Axis-Angle.
- **Vision**: Camera poses are usually $3 \times 3$ matrices or Quaternions.

## Theory & Background

### Representations
1.  **Rotation Matrix ($3 \times 3$)**: Unique, no singularities, but uses 9 parameters for 3 DoF.
2.  **Euler Angles (Roll, Pitch, Yaw)**: Intuitive, but suffers from **Gimbal Lock**. Order matters (XYZ vs ZYX).
3.  **Axis-Angle**: Rotation by angle $\theta$ around unit axis $u$.

### Eigen Geometry
- `Eigen::AngleAxisd(angle, axis)`
- `matrix.eulerAngles(0, 1, 2)` for XYZ.

## Implementation Tasks

### Task 1: Euler to Matrix
Convert Roll (X), Pitch (Y), Yaw (Z) to a Rotation Matrix using ZYX convention ($R = R_z R_y R_x$).

### Task 2: Matrix to Axis-Angle
Extract the rotation axis and angle from a matrix.

### Task 3: Matrix to Euler
Recover the angles. Note: This is ambiguous and hard to get right due to multiple solutions.

## Common Pitfalls
- **Gimbal Lock**: When Pitch is +/- 90 degrees.
- **Order**: Eigen's `eulerAngles(2, 1, 0)` corresponds to Z, Y, X axes.
- **Range**: Euler angles returned by Eigen are in $[-\pi, \pi]$.
