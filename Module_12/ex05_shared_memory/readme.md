# Exercise 05: Shared Memory (Matrix Multiplication)

## Goal
Optimize Matrix Multiplication ($C = A \times B$) using CUDA Shared Memory to reduce global memory bandwidth consumption.

## Learning Objectives
1.  Understand the memory hierarchy: Global (slow, large) vs. Shared (fast, small) vs. Registers.
2.  Implement a tiled matrix multiplication algorithm.
3.  Use `__shared__` to declare shared memory arrays.
4.  Use `__syncthreads()` to synchronize threads within a block.

## Practical Motivation
Naive matrix multiplication performs $2N^3$ memory accesses for $N^3$ compute operations. By loading a tile of data into shared memory (which functions as a user-managed cache), we can reuse data threads within the same block, significantly increasing the Compute-to-Memory ratio. This "tiling" pattern is ubiquitous in high-performance computing (convolution, GEMM).

## Theory: Tiling
We divide matrices $A, B, C$ into square tiles of size $TILE\_WIDTH \times TILE\_WIDTH$.
1.  Each thread block computes one tile of $C$.
2.  Each thread computes one element of $C$.
3.  The loop iterates over tiles of $A$ and $B$.
4.  **Phase 1:** Load a tile of $A$ and a tile of $B$ into Shared Memory.
5.  **Phase 2:** `__syncthreads()` to ensure all data is loaded.
6.  **Phase 3:** Compute partial product using cached data.
7.  **Phase 4:** `__syncthreads()` before loading the next tile.

## Step-by-Step Instructions

### Task 1: The Kernel (`src/matrix_mul.cu`)
1.  Define constants `TILE_WIDTH` (e.g., 16 or 32).
2.  Declare shared memory:
    ```cpp
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    ```
3.  Calculate row/col indices.
4.  Loop over phases `p` from `0` to `width/TILE_WIDTH`.
5.  Inside loop:
    -   Load `A[row][p*TILE_WIDTH + tx]` into `As[ty][tx]`.
    -   Load `B[p*TILE_WIDTH + ty][col]` into `Bs[ty][tx]`.
    -   Handle boundary checks if dimensions aren't multiples of tile size (for simplicity, assume they are or pad with 0).
    -   `__syncthreads()`.
    -   Accumulate dot product.
    -   `__syncthreads()`.
6.  Write result to `C`.

### Task 2: Host Code
1.  Initialize matrices $A$ (NxK) and $B$ (KxM).
2.  Launch kernel with `dim3 block(TILE_WIDTH, TILE_WIDTH)` and appropriate grid size.
3.  Verify against CPU implementation.

## Common Pitfalls
-   **Missing Synchronization:** Forgetting `__syncthreads()` leads to race conditions (reading data before it's written).
-   **Bank Conflicts:** Not a major issue for simple float32 matrix mul, but good to be aware of.
-   **Out of Bounds:** If matrix size is not a multiple of `TILE_WIDTH`, you must mask loads/stores.

## Code Hints
```cpp
__global__ void matrixMulShared(float* A, float* B, float* C, int w) {
    __shared__ float tile_A[WIDTH][WIDTH];
    __shared__ float tile_B[WIDTH][WIDTH];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0;

    for (int p = 0; p < w/WIDTH; ++p) {
        tile_A[threadIdx.y][threadIdx.x] = A[row * w + (p * WIDTH + threadIdx.x)];
        tile_B[threadIdx.y][threadIdx.x] = B[(p * WIDTH + threadIdx.y) * w + col];
        __syncthreads();
        
        for (int k = 0; k < WIDTH; ++k)
            val += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        __syncthreads();
    }
    C[row * w + col] = val;
}
```

## Verification
Compare GPU output with a simple 3-loop CPU matrix multiplication. Error should be negligible (~1e-5).
