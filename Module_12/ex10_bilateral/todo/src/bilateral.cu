#include "bilateral.h"
#include <cuda_runtime.h>
#include <iostream>

#define RADIUS 3
#define KERNEL_DIM (2 * RADIUS + 1)

// TODO: Constant memory declaration
// __constant__ float c_spatial[...];

// TODO: Kernel
// __global__ void bilateralFilterKernel(...) {
//     // Read center
//     // Loop neighbors
//     // Compute weights (Spatial from constant, Range from diff)
//     // Normalize
// }

void bilateralFilterWrapper(const unsigned char* d_in, unsigned char* d_out, int width, int height, float sigma_s, float sigma_r) {
    // TODO: Precompute Spatial Weights on Host
    
    // TODO: Copy to Constant Memory
    
    // TODO: Launch Kernel
}
