# Exercise 06: Mahalanobis Distance (Gating)

## Goal
Implement Mahalanobis Distance calculation to reject outliers (Gating) in a tracking system.

## Learning Objectives
1.  Understand the difference between Euclidean and Mahalanobis distance.
2.  Calculate Mahalanobis distance using Innovation Covariance $S$.
3.  Implement a Chi-Squared gating check.

## Practical Motivation
In tracking, we predict where a measurement should fall ($z_{pred}$ with covariance $S$). If a real measurement $z$ is far from $z_{pred}$ *relative to the uncertainty $S$*, it is likely an outlier or clutter, not the object we are tracking. Euclidean distance ignores the shape of uncertainty (covariance ellipse). Mahalanobis distance accounts for it.

## Theory
$$ d_M^2 = (z - \hat{z})^T S^{-1} (z - \hat{z}) $$
If $d_M^2 > \chi^2_{threshold}$, reject the measurement.
For 2D measurement ($z \in R^2$), $\chi^2_{0.95} \approx 5.99$.

## Step-by-Step Instructions

### Task 1: Setup
1.  Define a predicted measurement `z_pred` (e.g., [0,0]).
2.  Define covariance `S` (e.g., diagonal [2, 0.5]). This means high uncertainty in X, low in Y.
3.  Define a list of candidate measurements.

### Task 2: Implementation
1.  Function `computeMahalanobis(z, z_pred, S)`.
2.  Function `isGated(z, z_pred, S, threshold)`.

### Task 3: Visualization/Test
1.  Test with a point at (2,0). Euclidean dist = 2. Mahalanobis might be small because variance in X is 2.
2.  Test with a point at (0,2). Euclidean dist = 2. Mahalanobis will be huge because variance in Y is 0.5.
3.  Print results.

## Common Pitfalls
-   **Inversion:** $S$ must be invertible. Add small epsilon to diagonal if needed.
-   **Threshold:** Choose based on degrees of freedom (dim of z) and desired confidence (e.g., 95%, 99%).

## Code Hints
```cpp
double d2 = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
return d2;
```

## Verification
Points aligned with the major axis of the covariance ellipse should be accepted even if far away. Points aligned with minor axis should be rejected even if closer.
