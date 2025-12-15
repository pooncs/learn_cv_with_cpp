# Exercise 10: Bilateral Filter (Optimized Kernel)

## Goal
Implement a Bilateral Filter kernel from scratch. This filter smoothes images while preserving edges, unlike Gaussian blur.

## Learning Objectives
1.  Understand the Bilateral Filter math (Spatial weight + Range weight).
2.  Implement a non-linear filter where weights depend on pixel values.
3.  Optimize using Constant Memory for spatial weights (Gaussian kernel).
4.  Optimize using pre-computed look-up tables (LUT) for range weights (optional).

## Practical Motivation
Bilateral filtering is crucial for denoising without losing edge details. It is computationally expensive ($O(R^2)$ per pixel with complex exp calculations). GPU acceleration is highly effective here.

## Theory: Bilateral Filter
$$ I_{filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(||I(x_i) - I(x)||) g_s(||x_i - x||) $$
-   $g_s$: Spatial Gaussian (distance based).
-   $f_r$: Range Gaussian (intensity difference based).
-   $W_p$: Normalization factor.

## Step-by-Step Instructions

### Task 1: Constant Memory (`src/bilateral.cu`)
1.  Precompute the spatial Gaussian weights on the host.
2.  Copy them to `__constant__` memory on the GPU.
    -   `__constant__ float c_gaussian[RADIUS*2+1][RADIUS*2+1];`
    -   `cudaMemcpyToSymbol(...)`

### Task 2: The Kernel
1.  Load center pixel $I(x)$.
2.  Loop over window (neighbor $x_i$).
3.  Read $I(x_i)$.
4.  Compute range difference $diff = I(x_i) - I(x)$.
5.  Compute range weight $w_r = \exp(-diff^2 / (2\sigma_r^2))$.
6.  Read spatial weight $w_s$ from constant memory.
7.  Accumulate $sum += I(x_i) \cdot w_r \cdot w_s$.
8.  Accumulate weight $W += w_r \cdot w_s$.
9.  Normalize result.

### Task 3: Host Code
1.  Load noisy image.
2.  Launch kernel.
3.  Verify edge preservation (edges remain sharp).

## Common Pitfalls
-   **Register Pressure:** Complex math (exp) uses many registers.
-   **Texture Memory:** For reading input image, Texture Objects can be faster due to cache locality and boundary handling (clamp), but global memory is fine for this exercise.

## Code Hints
```cpp
__constant__ float c_spatial[KERNEL_SIZE];

// Kernel
float pixel = input[idx];
float w_range = expf(-(pixel - neighbor) * (pixel - neighbor) / (2 * sigma_r * sigma_r));
```

## Verification
Output image should look "smooth" in flat areas but sharp at edges (cartoon-like effect).
