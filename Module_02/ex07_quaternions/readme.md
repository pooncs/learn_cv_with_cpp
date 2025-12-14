# Exercise 07: Quaternions and SLERP

## Goal
Implement robust rotation interpolation (SLERP) using Eigen Quaternions.

## Learning Objectives
1.  Understand Quaternions as a rotation representation ($w, x, y, z$).
2.  Perform Spherical Linear Interpolation (SLERP) between two rotations.
3.  Avoid common pitfalls like antipodal ambiguity (q and -q represent same rotation).

## Practical Motivation
When animating a camera or robot end-effector from orientation A to B, linear interpolation of Euler angles or matrices produces bad results (speed changes, non-shortest path). SLERP provides a smooth, constant-speed interpolation along the shortest arc on the 4D hypersphere.

## Theory & Background

### Quaternion Basics
A unit quaternion $q = w + xi + yj + zk$ represents a rotation.
- **Unit Norm**: $||q|| = 1$.
- **Composition**: $q_{combined} = q_2 * q_1$.
- **Rotation**: $v_{rotated} = q * v * q^{-1}$.

### SLERP (Spherical Linear Interpolation)
$$ \text{Slerp}(q_1, q_2, t) = \frac{\sin((1-t)\Omega)}{\sin(\Omega)} q_1 + \frac{\sin(t\Omega)}{\sin(\Omega)} q_2 $$
where $\cos(\Omega) = q_1 \cdot q_2$.

## Implementation Tasks

### Task 1: Create Quaternions
Create two quaternions from Axis-Angle or Euler angles.

### Task 2: Implement SLERP
Use `Eigen::Quaternion::slerp` (or implement manually if you want a challenge, but using library function is standard).
Verify that at $t=0.5$, the rotation is halfway.

## Common Pitfalls
- **Normalization**: Quaternions must be normalized to represent valid rotations.
- **Double Cover**: $q$ and $-q$ are the same rotation. When interpolating, if $q_1 \cdot q_2 < 0$, negate one of them to take the shortest path. Eigen's `slerp` usually handles this.
