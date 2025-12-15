#include "bilateral.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define RADIUS 3
#define KERNEL_DIM (2 * RADIUS + 1)

// Constant memory for spatial weights
__constant__ float c_spatial[KERNEL_DIM * KERNEL_DIM];

__global__ void bilateralFilterKernel(const unsigned char* input, unsigned char* output, int width, int height, float two_sigma_r_sq) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int centerIdx = y * width + x;
    float centerVal = (float)input[centerIdx];
    
    float sumVal = 0.0f;
    float sumW = 0.0f;

    for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
        for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
            int nx = x + kx;
            int ny = y + ky;

            // Clamp to border
            nx = max(0, min(nx, width - 1));
            ny = max(0, min(ny, height - 1));

            float neighborVal = (float)input[ny * width + nx];
            
            // Spatial Weight (from Constant Memory)
            int sIdx = (ky + RADIUS) * KERNEL_DIM + (kx + RADIUS);
            float w_s = c_spatial[sIdx];

            // Range Weight (compute on fly)
            float diff = centerVal - neighborVal;
            float w_r = expf(-(diff * diff) / two_sigma_r_sq);

            float w = w_s * w_r;
            sumVal += neighborVal * w;
            sumW += w;
        }
    }

    output[centerIdx] = (unsigned char)(sumVal / sumW);
}

void bilateralFilterWrapper(const unsigned char* d_in, unsigned char* d_out, int width, int height, float sigma_s, float sigma_r) {
    // 1. Precompute Spatial Weights
    int kDim = 2 * RADIUS + 1;
    std::vector<float> h_spatial(kDim * kDim);
    float two_sigma_s_sq = 2.0f * sigma_s * sigma_s;
    
    for (int y = -RADIUS; y <= RADIUS; ++y) {
        for (int x = -RADIUS; x <= RADIUS; ++x) {
            float r2 = (float)(x*x + y*y);
            h_spatial[(y + RADIUS) * kDim + (x + RADIUS)] = expf(-r2 / two_sigma_s_sq);
        }
    }

    // 2. Copy to Constant Memory
    cudaMemcpyToSymbol(c_spatial, h_spatial.data(), h_spatial.size() * sizeof(float));

    // 3. Launch
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    float two_sigma_r_sq = 2.0f * sigma_r * sigma_r;
    bilateralFilterKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height, two_sigma_r_sq);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}
