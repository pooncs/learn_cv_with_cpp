# Exercise 04: Extended Kalman Filter (EKF)

## Goal
Implement an Extended Kalman Filter (EKF) to track a robot moving in a circle (Non-linear motion model) and measuring range/bearing (Non-linear measurement model).

## Learning Objectives
1.  Understand why Linear KF fails for non-linear systems.
2.  Implement the Jacobian calculation for $F$ (motion Jacobian) and $H$ (measurement Jacobian).
3.  Apply the EKF equations.

## Practical Motivation
Most real-world systems are non-linear.
-   **Motion:** Differential Drive robot ($x, y, \theta$).
-   **Sensors:** Radar/Lidar gives range ($r$) and bearing ($\phi$), not Cartesian $(x,y)$.
    $r = \sqrt{x^2 + y^2}$, $\phi = \arctan(y/x)$.

**Analogy:** Think of walking in a dark room. You know roughly how much you stepped (prediction), and you touch a wall (measurement). A Linear Kalman Filter assumes the walls are always straight and you walk in straight lines. An Extended Kalman Filter allows for curved walls and walking in circles by approximating them as small straight lines (tangents) at each step.

## Theory: EKF Equations
**State:** $x_k = f(x_{k-1}, u_k) + w_k$
**Measurement:** $z_k = h(x_k) + v_k$

1.  **Predict:**
    $$ \hat{x}^- = f(\hat{x}_{k-1}, u_k) $$
    $$ P^- = F P F^T + Q $$
    Where $F = \frac{\partial f}{\partial x} |_{\hat{x}_{k-1}}$ (Jacobian of motion).

2.  **Update:**
    $$ y = z - h(\hat{x}^-) $$ (Innovation)
    $$ S = H P^- H^T + R $$
    $$ K = P^- H^T S^{-1} $$
    $$ \hat{x} = \hat{x}^- + K y $$
    $$ P = (I - K H) P^- $$
    Where $H = \frac{\partial h}{\partial x} |_{\hat{x}^-}$ (Jacobian of measurement).

## Step-by-Step Instructions

### Task 1: Robot Model (CTRV or Simple Non-Linear)
Let's use a standard "Constant Turn Rate and Velocity" (CTRV) or simpler:
State: $[x, y, \theta]^T$.
Control: $[v, \omega]^T$ (linear vel, angular vel).
$$
x_{k+1} = x_k + v \cos(\theta_k) \Delta t \\
y_{k+1} = y_k + v \sin(\theta_k) \Delta t \\
\theta_{k+1} = \theta_k + \omega \Delta t
$$

Jacobian $F$:
$$
F = \begin{bmatrix}
1 & 0 & -v \sin(\theta)\Delta t \\
0 & 1 & v \cos(\theta)\Delta t \\
0 & 0 & 1
\end{bmatrix}
$$

### Task 2: Measurement Model (Range-Bearing)
Measurement $z = [r, \phi]^T$.
Landmark at $(0,0)$.
$$
r = \sqrt{x^2 + y^2} \\
\phi = \arctan2(y, x) - \theta
$$ (Bearing relative to robot heading)

Jacobian $H$:
$$
H = \begin{bmatrix}
\frac{x}{\sqrt{x^2+y^2}} & \frac{y}{\sqrt{x^2+y^2}} & 0 \\
\frac{-y}{x^2+y^2} & \frac{x}{x^2+y^2} & -1
\end{bmatrix}
$$

### Task 3: Implementation
1.  Define `EKF` class using Eigen.
2.  Implement `predict(u)` calculating $F$ dynamically.
3.  Implement `update(z)` calculating $H$ dynamically.
    *Note: Normalize angles in innovation $y$ to $[-\pi, \pi]$.*

### Task 4: Simulation
1.  Robot drives in a circle ($v=1, \omega=0.1$).
2.  Measure range/bearing to landmark at origin (or fixed point).
3.  Compare EKF estimate with Ground Truth.

## Common Pitfalls
-   **Angle Wrapping:** $\phi - \hat{\phi}$ can result in $2\pi$ errors. Always normalize angles.
-   **Jacobian Derivation:** Double check the math.
-   **Singularities:** If $r=0$, Jacobian is undefined. (Robot on top of landmark).

## Code Hints
```cpp
// Normalize Angle
while (y(1) > M_PI) y(1) -= 2 * M_PI;
while (y(1) < -M_PI) y(1) += 2 * M_PI;
```

## Verification
The robot position should trace a circle. The covariance ellipse should align with the tangential direction (uncertainty grows along the path).
