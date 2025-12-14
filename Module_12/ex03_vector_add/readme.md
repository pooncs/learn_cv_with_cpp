# Exercise 03: Vector Addition

## Goal
Write your first CUDA Kernel to add two large arrays in parallel.

## Learning Objectives
1.  Understand the CUDA programming model (Host vs Device code).
2.  Write a kernel function using the `__global__` qualifier.
3.  Understand the execution configuration `<<<grid, block>>>`.
4.  Calculate global thread indices.
5.  Handle array sizes that are not multiples of the block size.

## Practical Motivation
Vector addition is the "Hello World" of parallel computing. While adding two arrays on a CPU happens sequentially (looping 0 to N-1), a GPU can launch N threads to perform the additions simultaneously. This is the basis for more complex operations like image brightness adjustment (adding a constant to every pixel).

## Theory: Thread Hierarchy
-   **Grid:** A collection of blocks.
-   **Block:** A collection of threads.
-   **Thread:** The smallest unit of execution.

To find the unique index of a thread in a 1D grid:
$$ i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x} $$

## Step-by-Step Instructions

### Task 1: The Kernel (`src/vector_add.cu`)
1.  Define a function with `__global__` qualifier:
    ```cpp
    __global__ void vectorAddKernel(const float* A, const float* B, float* C, int N);
    ```
2.  Inside the kernel:
    -   Calculate global index `i`.
    -   **Boundary Check:** Ensure `i < N`. Since we launch blocks, total threads might exceed N.
    -   Perform addition: `C[i] = A[i] + B[i];`.

### Task 2: The Host Launcher (`src/vector_add.cu`)
1.  Implement a host function `vectorAdd(const float* d_A, const float* d_B, float* d_C, int N)`.
2.  Define block size (e.g., 256 threads).
3.  Calculate grid size:
    -   We need enough blocks to cover N elements.
    -   `gridSize = (N + blockSize - 1) / blockSize;` (Ceiling division).
4.  Launch the kernel: `vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);`.
5.  Check for launch errors using `cudaGetLastError()` and `cudaDeviceSynchronize()`.

### Task 3: Main Application (`src/main.cpp`)
1.  Allocate host vectors A, B, C. Initialize A and B with random data.
2.  Allocate device memory.
3.  Copy A and B to device.
4.  Call `vectorAdd`.
5.  Copy C back to host.
6.  Verify result against CPU implementation.

## Common Pitfalls
-   **Missing Boundary Check:** If you launch 1024 threads for 1000 elements, threads 1000-1023 will access out-of-bounds memory if you don't check `if (i < N)`.
-   **Integer Division:** `N / blockSize` truncates. Use `(N + blockSize - 1) / blockSize` for ceiling.
-   **Kernel Asynchrony:** Kernels return immediately. `cudaMemcpy` (HostToDevice or DeviceToHost) implicitly synchronizes, but if you want to catch kernel errors immediately, use `cudaDeviceSynchronize()`.

## Code Hints
```cpp
// Kernel
__global__ void add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Launch
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

## Verification
The program prints "PASS" if the maximum error between GPU and CPU result is below a small epsilon (e.g., 1e-5).
