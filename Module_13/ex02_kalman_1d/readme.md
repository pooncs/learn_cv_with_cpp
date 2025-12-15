# Exercise 02: Kalman Filter 1D (Constant Voltage)

## Goal
Implement a 1D Kalman Filter to track a constant voltage signal corrupted by Gaussian noise.

## Learning Objectives
1.  Understand the Kalman Filter equations (Predict & Update).
2.  Implement the scalar version of KF.
3.  Visualize how the filter estimates state $x$ and covariance $P$ over time.

## Practical Motivation
This is the simplest KF case. Imagine reading a voltmeter connected to a battery. The true voltage is constant (e.g., 1.25V), but the sensor is noisy ($1.20, 1.30, 1.22...$). A KF combines the "Constant Model" (prediction) with the "Measurement" to find the optimal estimate.

## Theory: 1D KF Equations
**State:** $x$ (Voltage)
**Model:** Constant ($x_k = x_{k-1} + 0$)
**Measurement:** $z_k = x_k + v_k$

1.  **Predict:**
    $$ \hat{x}_{k}^- = \hat{x}_{k-1} $$
    $$ P_{k}^- = P_{k-1} + Q $$ (Q: Process Noise Variance)

2.  **Update:**
    $$ K_k = P_k^- (P_k^- + R)^{-1} $$ (Kalman Gain)
    $$ \hat{x}_k = \hat{x}_k^- + K_k (z_k - \hat{x}_k^-) $$
    $$ P_k = (1 - K_k) P_k^- $$

## Step-by-Step Instructions

### Task 1: Setup
1.  Define a class `KalmanFilter1D`.
2.  Members: `x` (state), `P` (covariance), `Q` (process noise), `R` (measurement noise).
3.  Constructor: Initialize `x=0`, `P=1.0`, `Q=1e-5`, `R=0.1` (example values).

### Task 2: Implement Methods
1.  `predict()`: Update `P = P + Q`. State remains same.
2.  `update(double measurement)`: Calculate K, update x, update P.

### Task 3: Simulation
1.  True Voltage = 1.25.
2.  Generate 100 noisy measurements (add random Gaussian noise).
3.  Run KF loop:
    -   Predict
    -   Update(z)
    -   Print `z` vs `x` vs `True`.
4.  Observe how `x` converges to 1.25 and `P` decreases.

## Common Pitfalls
-   **Initial P:** If P is too small (confident), it takes longer to converge. If too large, it jumps to the first measurement.
-   **Q vs R:** Tuning these is the "art" of KF.
    -   High Q -> Trust measurement (system changes fast).
    -   High R -> Trust model (sensor is noisy).

## Code Hints
```cpp
void update(double z) {
    double K = P / (P + R);
    x = x + K * (z - x);
    P = (1 - K) * P;
}
```

## Verification
The estimated `x` should be closer to 1.25 than the raw measurements `z` (lower standard deviation).
