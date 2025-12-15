#include "matrix_mul.h"
#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16

__global__ void matrixMulSharedKernel(const float* A, const float* B, float* C, int width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float val = 0.0f;

    // Loop over tiles
    for (int p = 0; p < width / TILE_WIDTH; ++p) {
        // Load tile from A into Shared Memory
        As[ty][tx] = A[row * width + (p * TILE_WIDTH + tx)];
        
        // Load tile from B into Shared Memory
        Bs[ty][tx] = B[(p * TILE_WIDTH + ty) * width + col];

        // Wait for all threads to load
        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            val += As[ty][k] * Bs[k][tx];
        }

        // Wait before loading next tile
        __syncthreads();
    }

    if (row < width && col < width) {
        C[row * width + col] = val;
    }
}

void matrixMulWrapper(const float* d_A, const float* d_B, float* d_C, int width) {
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(width / TILE_WIDTH, width / TILE_WIDTH);

    matrixMulSharedKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}
