# Exercise 05: Unscented Kalman Filter (UKF)

## Goal
Implement an Unscented Kalman Filter (UKF) to solve the same non-linear tracking problem as Ex04. Compare UKF vs EKF.

## Learning Objectives
1.  Understand the "Unscented Transform" (Sigma Points).
2.  Implement Sigma Point generation.
3.  Implement Sigma Point Prediction and Update.
4.  Avoid calculating Jacobians.

## Practical Motivation
The EKF linearizes non-linear functions (Taylor Series). This approximation fails if the non-linearity is strong. The UKF picks specific sample points ("Sigma Points"), transforms them through the non-linear function, and recalculates the mean/covariance. It is often more accurate and easier to implement (no Jacobians!) but computationally heavier ($2N+1$ points).

## Theory: UKF Steps
1.  **Generate Sigma Points ($\chi$):**
    $$ \chi_0 = x $$
    $$ \chi_i = x + (\sqrt{(n+\lambda)P})_i \quad i=1..n $$
    $$ \chi_{i+n} = x - (\sqrt{(n+\lambda)P})_i \quad i=1..n $$
2.  **Predict Sigma Points:**
    $$ \chi_k^- = f(\chi_{k-1}, u) $$
3.  **Predicted Mean/Cov:**
    $$ \hat{x}^- = \sum w_i \chi_i^- $$
    $$ P^- = \sum w_i (\chi_i^- - \hat{x}^-)(\chi_i^- - \hat{x}^-)^T + Q $$
4.  **Update:**
    Transform points to measurement space: $Z_i = h(\chi_i^-)$.
    Calculate mean $\hat{z}$ and covariance $S$.
    Calculate Cross-Covariance $T$.
    $K = T S^{-1}$.
    Update state and covariance.

## Step-by-Step Instructions

### Task 1: Sigma Point Generation
1.  Define `lambda`, `n` (state dim).
2.  Compute matrix square root `L = P.llt().matrixL()`.
3.  Create $2n+1$ vectors.

### Task 2: UKF Class
1.  `predict()`:
    -   Generate Sigma Points.
    -   Pass each through process model `f(x)`.
    -   Reconstruct Mean/Cov.
2.  `update()`:
    -   Pass predicted Sigma Points through measurement model `h(x)`.
    -   Compute predicted measurement mean $\hat{z}$.
    -   Compute $S$ (innovation cov) and $T$ (cross cov).
    -   Update.

### Task 3: Comparison
Run the same simulation as Ex04.
Check if UKF is more robust or accurate (especially for high non-linearity or large time steps).

## Common Pitfalls
-   **Matrix Square Root:** Cholesky decomposition requires Positive Definite matrix. If $P$ becomes non-PD due to numerical errors, UKF crashes.
-   **Angle Averaging:** When averaging angles (e.g., yaw), simple arithmetic mean fails (average of $10^\circ$ and $350^\circ$ is $180^\circ$, should be $0^\circ$). Use vector sums or normalize diffs.

## Code Hints
```cpp
// Weights
weights(0) = lambda / (n + lambda);
for (i=1..2n) weights(i) = 0.5 / (n + lambda);

// Angle Diff in Covariance
VectorXd diff = X.col(i) - x;
while (diff(2) > PI) diff(2) -= 2*PI;
// ...
```

## Verification
UKF should track the robot effectively. In this simple case, performance might be similar to EKF, but code should be Jacobian-free.
