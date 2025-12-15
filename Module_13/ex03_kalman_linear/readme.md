# Exercise 03: Linear Kalman Filter (CV Model)

## Goal
Implement a generic Linear Kalman Filter using Eigen and apply it to a 2D Constant Velocity (CV) tracking problem.

## Learning Objectives
1.  Use `Eigen` library for matrix operations ($A, P, Q, R, K$).
2.  Define state vectors ($x = [p_x, p_y, v_x, v_y]^T$) and transition matrices.
3.  Track a 2D moving object (e.g., a mouse cursor or simulated point).

## Practical Motivation
Tracking objects in 2D images often assumes they move smoothly. The Constant Velocity (CV) model is a workhorse for tracking pedestrians, vehicles, etc., over short time intervals.

## Theory: CV Model
**State:** $x = [p_x, p_y, v_x, v_y]^T$
**Transition Matrix ($A$):**
$$
A = \begin{bmatrix} 
1 & 0 & \Delta t & 0 \\ 
0 & 1 & 0 & \Delta t \\ 
0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}
$$
**Measurement Matrix ($H$):** We observe position only.
$$
H = \begin{bmatrix} 
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 
\end{bmatrix}
$$

**KF Equations (Matrix Form):**
1.  Predict:
    $$ \hat{x}^- = A \hat{x} $$
    $$ P^- = A P A^T + Q $$
2.  Update:
    $$ K = P^- H^T (H P^- H^T + R)^{-1} $$
    $$ \hat{x} = \hat{x}^- + K (z - H \hat{x}^-) $$
    $$ P = (I - K H) P^- $$

## Step-by-Step Instructions

### Task 1: Generic KF Class
Create a class `KalmanFilter` that holds Eigen matrices.
-   `init(x0, P0, A, H, Q, R)`
-   `predict()`
-   `update(z)`

### Task 2: CV Model Setup
In `main`:
1.  Define $\Delta t = 0.1$.
2.  Construct $A$ (4x4), $H$ (2x4).
3.  Construct $Q$ (Process noise). Assume small acceleration noise.
4.  Construct $R$ (Measurement noise). Assume GPS-like noise ($\sigma^2 \approx 0.1$).

### Task 3: Simulation
1.  Ground Truth: Object moving at constant velocity $(1, 0.5)$.
    $x_t = 1 \cdot t, y_t = 0.5 \cdot t$.
2.  Measurements: Add noise to true $x, y$.
3.  Run KF loop.

## Common Pitfalls
-   **Eigen Initialization:** Matrices default to uninitialized or zero. Ensure $P$ is Identity or large diagonal.
-   **Matrix Dimensions:** Mismatch between 4x4 and 2x4 matrices is a common compile-time or run-time error in Eigen.

## Code Hints
```cpp
// Predict
x = A * x;
P = A * P * A.transpose() + Q;

// Update
Eigen::MatrixXd S = H * P * H.transpose() + R;
Eigen::MatrixXd K = P * H.transpose() * S.inverse();
x = x + K * (z - H * x);
P = (Eigen::MatrixXd::Identity(4,4) - K * H) * P;
```

## Verification
The KF velocity estimate ($v_x, v_y$) should converge to $(1, 0.5)$ even though we only measure position.
