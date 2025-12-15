# Exercise 08: Thrust Library

## Goal
Use the Thrust library to perform high-level parallel algorithms (sort, reduce, transform) without writing custom kernels.

## Learning Objectives
1.  Understand Thrust's container system: `thrust::host_vector` vs. `thrust::device_vector`.
2.  Use `thrust::generate` to fill data.
3.  Use `thrust::sort` for parallel sorting.
4.  Use `thrust::transform` with custom functors (lambdas) for element-wise operations.
5.  Use `thrust::reduce` to compute sum/min/max.

## Practical Motivation
Writing custom kernels for reduction or sorting is complex and error-prone. Thrust provides STL-like algorithms that are highly optimized for NVIDIA GPUs. It allows rapid prototyping and is often faster than naive custom kernels.

## Theory: Functors
Thrust relies on C++ functors (structs with `operator()`) or lambdas (in modern CUDA) to define custom operations passed to algorithms.
```cpp
struct Square {
    __host__ __device__
    float operator()(const float& x) const {
        return x * x;
    }
};
```

## Step-by-Step Instructions

### Task 1: Data Setup (`src/thrust_demo.cu`)
1.  Create `thrust::host_vector<float>` of size N (e.g., 1M).
2.  Fill with random data using `thrust::generate` or `std::generate`.
3.  Copy to `thrust::device_vector<float>`.

### Task 2: Algorithms
1.  **Sort:** `thrust::sort(d_vec.begin(), d_vec.end());`.
2.  **Transform:** Square every element.
    -   Define a `Square` functor.
    -   `thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), Square());`.
3.  **Reduce:** Compute sum.
    -   `float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());`.

### Task 3: Verification
1.  Copy result back to host.
2.  Verify sorting and sum (approximate check for sum due to float precision).

## Common Pitfalls
-   **Namespace:** Algorithms are in `thrust::` namespace.
-   **Device vs Host iterators:** Mixing iterators (e.g., passing host iterator to device sort) causes compilation errors or crashes.
-   **Lambda support:** Requires `--expt-extended-lambda` flag in `nvcc` for complex lambdas, though simple functors work everywhere.

## Code Hints
```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// ...
thrust::device_vector<float> d_vec = h_vec;
thrust::sort(d_vec.begin(), d_vec.end());
```

## Verification
Output "PASS" if the vector is sorted and the sum matches CPU calculation.
