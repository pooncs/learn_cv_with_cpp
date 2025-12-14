# Exercise 04: Linear Solvers

## Goal
Solve systems of linear equations $Ax=b$ using various decompositions (LLT, LDLT, LU).

## Learning Objectives
1.  Understand when to use different solvers based on matrix properties (SPD, Positive Definite, General).
2.  Implement solvers for $Ax=b$ in Eigen.
3.  Compare accuracy and performance (conceptually).

## Practical Motivation
In Computer Vision, we solve linear systems constantly:
- **Camera Calibration**: Solving for intrinsic parameters.
- **Bundle Adjustment**: Solving normal equations $J^T J \Delta x = -J^T r$.
- **Optical Flow**: Solving $A v = b$ for velocity vector $v$.

## Theory & Background

### Choice of Decompositions
1.  **LLT (Cholesky)**:
    - Requirements: Matrix $A$ must be **Symmetric Positive Definite (SPD)**.
    - Speed: Very fast.
    - Accuracy: Good.
2.  **LDLT (Robust Cholesky)**:
    - Requirements: Symmetric, Positive or Negative Semidefinite.
    - Robustness: More robust than LLT if matrix is near singular.
3.  **HouseholderQR**:
    - Requirements: Any matrix.
    - Speed: Slower but very stable.
4.  **Lu (FullPivLU, PartialPivLU)**:
    - Requirements: Square, invertible.

## Implementation Tasks

### Task 1: LLT Solver
Given an SPD matrix $A$ and vector $b$, solve for $x$ using `A.llt().solve(b)`.

### Task 2: LDLT Solver
Solve for a slightly less well-behaved symmetric matrix using `A.ldlt().solve(b)`.

### Task 3: Check Error
Compute the relative error $||Ax - b|| / ||b||$.

## Common Pitfalls
- Trying to use LLT on a non-SPD matrix (result will be numerical garbage or NaN).
- Forgetting to check if the decomposition succeeded (`info() == Eigen::Success`).
