# Exercise 10: Custom Filter

## Goal
Build an optimized separable Gaussian filter.

## Learning Objectives
1.  Understand 2D convolution as a sequence of two 1D convolutions (Separability).
2.  Implement a 1D Gaussian kernel generator.
3.  Optimize filtering performance.

## Practical Motivation
A $N \times N$ convolution requires $N^2$ ops per pixel. Separable convolution requires $2N$ ops. For large kernels (e.g., $15 \times 15$), this is a massive speedup ($225$ vs $30$ ops). Gaussian blur is separable.

## Theory & Background

### Gaussian Kernel
$$ G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}} $$

### Separability
Convolution is associative.
$$ I * G_{2D} = I * (G_{1D}^T * G_{1D}) = (I * G_{1D}^T) * G_{1D} $$
We filter rows first, then filter columns (or vice versa).

## Implementation Tasks

### Task 1: Generate Kernel
Implement `get_gaussian_kernel(int size, double sigma)` returning a $1 \times N$ matrix.

### Task 2: Separable Filter
Implement `separable_filter(src, kernelX, kernelY)`.
1.  Filter rows with `kernelX`.
2.  Filter columns with `kernelY`.

## Common Pitfalls
- Kernel size should be odd.
- Normalize the kernel so sum is 1.0 (to preserve brightness).
