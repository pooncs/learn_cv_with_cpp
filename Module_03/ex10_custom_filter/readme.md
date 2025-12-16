# Exercise 10: Custom Filter (Separable)

## Goal
Build an optimized separable Gaussian filter.

## Learning Objectives
1.  **Separability:** Decomposing a 2D matrix into two 1D vectors ($K = v \times h^T$).
2.  **Optimization:** Why $2N$ operations is better than $N^2$.
3.  **Gaussian Kernel:** The bell curve formula.

## Analogy: The Efficient Painter
*   **The Job:** You need to blur a picture using a $11 \times 11$ brush.
*   **Naive Approach (2D Convolution):** For every pixel, you mix colors from 121 neighbors ($11^2$).
    *   *Cost:* 121 operations per pixel. Slow.
*   **Separable Approach:**
    *   **Pass 1:** You only blur **Horizontally** ($1 \times 11$). Cost: 11 ops.
    *   **Pass 2:** You take that result and blur **Vertically** ($11 \times 1$). Cost: 11 ops.
    *   **Total Cost:** 22 ops per pixel.
    *   *Result:* Mathematically identical to the 2D blur, but **5.5x faster**.

## Practical Motivation
*   **Speed:** Gaussian Blur, Sobel, and Box Filters are all separable. OpenCV uses this optimization internally.
*   **Large Kernels:** The benefit grows with kernel size. For $31 \times 31$, separability makes it 15x faster.

## Step-by-Step Instructions

### Task 1: Generate Kernel
Open `src/main.cpp`.
*   Implement `get_gaussian_kernel(int ksize, double sigma)`.
*   Formula: $G(x) = \exp(-\frac{(x - center)^2}{2\sigma^2})$.
*   Normalize: Sum of elements must be 1.0.

### Task 2: Separable Filter
*   Implement `custom_separable_filter(Mat& src, Mat& dst, int ksize, double sigma)`.
*   Generate the 1D kernel.
*   **Pass 1:** Filter rows (`cv::filter2D` with $1 \times N$ kernel).
*   **Pass 2:** Filter columns (`cv::filter2D` with $N \times 1$ kernel).

### Task 3: Compare
*   Compare result with `cv::GaussianBlur`.
*   Compare speed (optional).

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
The output should look exactly like standard Gaussian Blur.
