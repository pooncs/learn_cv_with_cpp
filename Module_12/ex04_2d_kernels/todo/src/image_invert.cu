#include "image_invert.h"
#include <cuda_runtime.h>
#include <iostream>

// TODO: Implement Kernel
// __global__ void invertImageKernel(...)

void invertImageWrapper(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    // TODO: Define Block and Grid dimensions
    // dim3 blockSize(..., ...);
    // dim3 gridSize(..., ...);

    // TODO: Launch Kernel
    // invertImageKernel<<<...>>>(...);
}
