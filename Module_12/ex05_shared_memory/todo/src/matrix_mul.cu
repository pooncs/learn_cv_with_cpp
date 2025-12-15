#include "matrix_mul.h"
#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16

// TODO: Implement Shared Memory Matrix Multiplication
// __global__ void matrixMulSharedKernel(...)

void matrixMulWrapper(const float* d_A, const float* d_B, float* d_C, int width) {
    // TODO: Define Block and Grid
    // dim3 blockSize(..., ...);
    // dim3 gridSize(..., ...);

    // TODO: Launch
    // matrixMulSharedKernel<<<...>>>(...);
}
