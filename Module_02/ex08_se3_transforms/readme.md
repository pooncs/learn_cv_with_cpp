# Exercise 08: Rigid Body Transforms (SE3)

## Goal
Construct $4 \times 4$ transformation matrices (SE3) to chain robot joint poses.

## Learning Objectives
1.  Understand Homogeneous Coordinates.
2.  Construct SE(3) matrices from Rotation ($3 \times 3$) and Translation ($3 \times 1$).
3.  Apply transformations to 3D points.
4.  Chain transformations ($T_{world\_hand} = T_{world\_base} \times T_{base\_shoulder} \times \dots$).

## Practical Motivation
In Robotics and AR/VR, we represent the pose of objects using SE(3) matrices. Chaining them allows us to calculate the position of an end-effector relative to the world frame (Forward Kinematics).

## Theory & Background

### SE(3) Matrix
A rigid body transform is a member of the Special Euclidean group SE(3).
$$ T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} $$
where $R \in SO(3)$ is rotation, and $t \in \mathbb{R}^3$ is translation.

### Transforming Points
To transform a point $p = [x, y, z]^T$:
1.  Convert to homogeneous coordinates: $p_h = [x, y, z, 1]^T$.
2.  Multiply: $p'_h = T \times p_h$.
3.  Convert back: $p' = [p'_x, p'_y, p'_z]^T$.

## Implementation Tasks

### Task 1: Create SE3
Implement a function that takes a $3 \times 3$ rotation and $3 \times 1$ translation and returns a $4 \times 4$ matrix.

### Task 2: Apply Transform
Implement a function that applies $T$ to a 3D point $p$.

### Task 3: Chain Transforms
Create two transforms $T_1$ (Translation X+1) and $T_2$ (Rotation Z 90 deg). Compute $T_{combined} = T_1 \times T_2$ vs $T_2 \times T_1$.

## Common Pitfalls
- **Order of Multiplication**: Matrix multiplication is non-commutative. $T_1 T_2$ means apply $T_2$ first (in local frame) or $T_1$ first (in global frame), depending on convention (Pre-multiply vs Post-multiply). In standard column-vector notation $p' = T_2 T_1 p$, $T_1$ is applied first.
