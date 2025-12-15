#include "box_filter.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_W 16
#define BLOCK_H 16
#define RADIUS 1

__global__ void boxFilterKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    // Shared memory size: (16+2) x (16+2) = 18x18
    __shared__ unsigned char smem[BLOCK_H + 2 * RADIUS][BLOCK_W + 2 * RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    // Local index in shared memory (offset by radius)
    int lx = tx + RADIUS;
    int ly = ty + RADIUS;

    // 1. Load Center
    if (gx < width && gy < height)
        smem[ly][lx] = input[gy * width + gx];
    else
        smem[ly][lx] = 0;

    // 2. Load Halo (Simplified: only works for RADIUS=1, BLOCK >= RADIUS)
    // Top
    if (ty < RADIUS) {
        if (gy >= RADIUS) 
            smem[ly - RADIUS][lx] = input[(gy - RADIUS) * width + gx];
        else 
            smem[ly - RADIUS][lx] = 0;
    }
    // Bottom
    if (ty >= blockDim.y - RADIUS) {
        if (gy + RADIUS < height)
            smem[ly + RADIUS][lx] = input[(gy + RADIUS) * width + gx];
        else
            smem[ly + RADIUS][lx] = 0;
    }
    // Left
    if (tx < RADIUS) {
        if (gx >= RADIUS)
            smem[ly][lx - RADIUS] = input[gy * width + (gx - RADIUS)];
        else
            smem[ly][lx - RADIUS] = 0;
    }
    // Right
    if (tx >= blockDim.x - RADIUS) {
        if (gx + RADIUS < width)
            smem[ly][lx + RADIUS] = input[gy * width + (gx + RADIUS)];
        else
            smem[ly][lx + RADIUS] = 0;
    }
    
    // Corners (Top-Left, Top-Right, Bot-Left, Bot-Right)
    // Ideally we would loop or use a better loading strategy, but for RADIUS=1 explicit is fine
    if (tx < RADIUS && ty < RADIUS) { // Top-Left
        if (gx >= RADIUS && gy >= RADIUS) smem[ly-1][lx-1] = input[(gy-1)*width + gx-1];
        else smem[ly-1][lx-1] = 0;
    }
    if (tx >= blockDim.x - RADIUS && ty < RADIUS) { // Top-Right
        if (gx + RADIUS < width && gy >= RADIUS) smem[ly-1][lx+1] = input[(gy-1)*width + gx+1];
        else smem[ly-1][lx+1] = 0;
    }
    if (tx < RADIUS && ty >= blockDim.y - RADIUS) { // Bot-Left
        if (gx >= RADIUS && gy + RADIUS < height) smem[ly+1][lx-1] = input[(gy+1)*width + gx-1];
        else smem[ly+1][lx-1] = 0;
    }
    if (tx >= blockDim.x - RADIUS && ty >= blockDim.y - RADIUS) { // Bot-Right
        if (gx + RADIUS < width && gy + RADIUS < height) smem[ly+1][lx+1] = input[(gy+1)*width + gx+1];
        else smem[ly+1][lx+1] = 0;
    }

    __syncthreads();

    // 3. Compute
    if (gx < width && gy < height) {
        int sum = 0;
        for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
            for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                sum += smem[ly + dy][lx + dx];
            }
        }
        output[gy * width + gx] = (unsigned char)(sum / 9);
    }
}

void boxFilterWrapper(const unsigned char* d_in, unsigned char* d_out, int width, int height) {
    dim3 blockSize(BLOCK_W, BLOCK_H);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    boxFilterKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}
