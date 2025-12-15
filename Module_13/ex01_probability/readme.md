# Exercise 01: Probability Basics (Discrete Convolution)

## Goal
Implement a 1D Discrete Convolution to simulate the "Predict" step of a Bayes Filter (updating belief based on motion).

## Learning Objectives
1.  Understand the concept of a "Belief" distribution.
2.  Implement Total Probability theorem for motion updates:
    $$ \overline{bel}(x_t) = \sum_{x_{t-1}} P(x_t | u_t, x_{t-1}) \cdot bel(x_{t-1}) $$
3.  Visualize how uncertainty increases after motion (convolution spreads the peak).

## Practical Motivation
Before diving into Kalman Filters (which assume Gaussian distributions), it's crucial to understand the general mechanism of state estimation. When a robot moves, its position becomes uncertain. We represent this by convolving our current belief with a motion kernel (probability of moving distance $u$).

## Theory
-   **State Space:** 1D grid (e.g., 20 cells).
-   **Prior Belief:** Initial distribution (e.g., peak at index 10).
-   **Motion Model (Kernel):** Probability of moving $k$ steps. e.g., $P(move) = [0.1, 0.8, 0.1]$ (undershoot, exact, overshoot).
-   **Convolution:** The new belief at index $i$ is the sum of probabilities of coming from any previous index $j$.

## Step-by-Step Instructions

### Task 1: Setup
1.  Define a `std::vector<double> belief` of size 20. Initialize with a peak (e.g., index 10 = 1.0, others 0.0).
2.  Define a `std::vector<double> kernel` (e.g., `{0.1, 0.8, 0.1}`).

### Task 2: Implement Convolution
1.  Create a function `convolve(belief, kernel)`.
2.  Iterate over each position `i` in the new belief.
3.  For each `i`, sum over kernel indices `k`:
    `new_belief[i] += belief[i - k] * kernel[k]` (handle boundary conditions, e.g., cyclic or zero-padding).
    *Note: Standard convolution flips the kernel, but for motion "add", we often think in correlation. If kernel is symmetric, it's the same.*

### Task 3: Simulation
1.  Print initial belief.
2.  Apply convolution (Move).
3.  Print result. Notice the peak spreads (lower height, wider width).
4.  Apply again. Uncertainty grows.

## Common Pitfalls
-   **Normalization:** Ensure the kernel sums to 1.0. Ensure the resulting belief sums to 1.0 (if handling boundaries correctly).
-   **Indexing:** `i - k` can be negative. Handle cyclic world (modulo arithmetic) or clamp.

## Code Hints
```cpp
std::vector<double> convolve(const std::vector<double>& belief, const std::vector<double>& kernel) {
    int N = belief.size();
    int K = kernel.size();
    int offset = K / 2; // center of kernel
    std::vector<double> result(N, 0.0);
    
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            int idx = i - (k - offset);
            // Handle boundary (Cyclic)
            idx = (idx % N + N) % N; 
            result[i] += belief[idx] * kernel[k];
        }
    }
    return result;
}
```

## Verification
The sum of the belief vector should remain 1.0 (within float error). The peak should lower and spread after each step.
