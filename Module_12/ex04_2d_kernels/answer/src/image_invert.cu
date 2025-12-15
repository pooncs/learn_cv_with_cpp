#include "image_invert.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void invertImageKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; ++c) {
            output[idx + c] = 255 - input[idx + c];
        }
    }
}

void invertImageWrapper(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    invertImageKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}
