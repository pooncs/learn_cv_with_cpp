# Exercise 06: SIMD Intrinsics (AVX2)

## Goal
Optimize a computationally intensive operation (Dot Product) using Single Instruction, Multiple Data (SIMD) intrinsics. Specifically, we will use AVX2 to process 8 floats at once.

## Learning Objectives
1.  **Data Parallelism:** Understand how processing multiple data points with a single instruction speeds up code.
2.  **Intrinsics:** Learn to use `immintrin.h` functions like `_mm256_load_ps`, `_mm256_add_ps`, `_mm256_mul_ps`.
3.  **Memory Alignment:** Understand why SIMD often requires aligned memory.
4.  **Benchmarking:** Measure the speedup (typically 4x-8x) using Google Benchmark.

## Practical Motivation
In CV, operations like convolution, color conversion, and feature matching involve massive arrays of pixels. Scalar loops (processing one pixel at a time) leave the CPU's vector units idle. SIMD is key to achieving real-time performance.

## Theory: AVX2
AVX2 registers (`__m256`) are 256 bits wide, holding 8 `float`s (32 bits * 8 = 256).
*   `_mm256_loadu_ps(ptr)`: Load 8 floats from memory (unaligned).
*   `_mm256_add_ps(a, b)`: Add vector a and b.
*   `_mm256_fmadd_ps(a, b, c)`: Fused Multiply-Add: (a * b) + c.

## Step-by-Step Instructions

### Task 1: Scalar Implementation
Implement `dot_product_scalar(const float* a, const float* b, size_t n)`.
This is just a simple for-loop accumulating `sum += a[i] * b[i]`.

### Task 2: AVX2 Implementation
Implement `dot_product_avx2(const float* a, const float* b, size_t n)`.
1.  Loop with stride 8.
2.  Load 8 floats from `a` and `b`.
3.  Multiply and accumulate into a `sum_vec` register.
4.  Handle remaining elements (tail) if `n` is not divisible by 8.
5.  Horizontal sum: Sum the 8 elements inside `sum_vec` to get the final result.

### Task 3: Benchmark
Use Google Benchmark to compare the two implementations on a large array (e.g., 1M elements).

## Code Hints
*   **Horizontal Sum:** Summing a `__m256` is tricky. A simple way is to store it to a temporary array and sum that array, or use a sequence of hadd/permute instructions.
*   **Compilation:** Ensure you enable AVX2.
    *   MSVC: `/arch:AVX2`
    *   GCC/Clang: `-mavx2 -mfma`

## Verification
*   Check correctness: `scalar(a, b) == avx2(a, b)` (within epsilon).
*   Check performance: AVX2 should be significantly faster.
