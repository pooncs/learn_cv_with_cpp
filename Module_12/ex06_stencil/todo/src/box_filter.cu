#include "box_filter.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_W 16
#define BLOCK_H 16
#define RADIUS 1

// TODO: Implement Box Filter using Shared Memory
// __global__ void boxFilterKernel(...) {
//     // Define shared memory with Halo
//     // Load Center
//     // Load Halo
//     // Sync
//     // Compute Average
// }

void boxFilterWrapper(const unsigned char* d_in, unsigned char* d_out, int width, int height) {
    // TODO: Define Block and Grid
    // TODO: Launch
}
