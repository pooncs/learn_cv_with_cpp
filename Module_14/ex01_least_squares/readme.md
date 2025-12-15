# Exercise 01: Least Squares Optimization

## Goal
Implement the Gauss-Newton algorithm to solve a non-linear least squares problem: fitting a quadratic curve $y = ax^2 + bx + c$ to noisy data.

## Learning Objectives
1.  **Non-Linear Least Squares:** Understand the problem of minimizing $\sum ||f(x) - z||^2$.
2.  **Jacobian Matrix:** Learn how to compute the derivative of the residual with respect to parameters.
3.  **Gauss-Newton Algorithm:** Implement the iterative update rule $\Delta \theta = -(J^T J)^{-1} J^T r$.
4.  **Eigen:** Use the Eigen library for matrix operations.

## Practical Motivation
SLAM and Bundle Adjustment are essentially giant least squares problems. Before tackling those, you must understand how to optimize a simple model.

## Theory: Gauss-Newton
We want to find parameters $\theta = [a, b, c]^T$ that minimize the error between our model $f(x; \theta)$ and observations $y_{obs}$.
Residual $r_i = f(x_i; \theta) - y_{obs, i}$.
Cost $F(\theta) = \frac{1}{2} \sum r_i^2$.

The Gauss-Newton update is:
1.  Compute Residuals $r$.
2.  Compute Jacobian $J$ where $J_{ij} = \frac{\partial r_i}{\partial \theta_j}$.
3.  Solve Normal Equations: $(J^T J) \Delta \theta = -J^T r$.
4.  Update: $\theta \leftarrow \theta + \Delta \theta$.

## Step-by-Step Instructions

### Task 1: Generate Data
Generate points along $y = 1.0x^2 + 2.0x + 1.0$ with some random noise.

### Task 2: Define Model and Jacobian
For $f(x) = ax^2 + bx + c$:
*   $\frac{\partial r}{\partial a} = x^2$
*   $\frac{\partial r}{\partial b} = x$
*   $\frac{\partial r}{\partial c} = 1$

### Task 3: Implement Optimization Loop
Write a loop that runs for $N$ iterations or until convergence. Inside:
1.  Build $J$ (Nx3) and $r$ (Nx1).
2.  Solve for $\Delta \theta$.
3.  Apply update.

## Verification
*   Start with initial guess $a=0, b=0, c=0$.
*   After optimization, parameters should be close to $1.0, 2.0, 1.0$.
