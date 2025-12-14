#include "vector_add.h"
#include <cuda_runtime.h>
#include <iostream>

// TODO: Implement the kernel
// __global__ void vectorAddKernel(...) {
//     // Calculate global thread index
//     // Check bounds
//     // Add
// }

void vectorAddWrapper(const float* d_A, const float* d_B, float* d_C, int N) {
    // TODO: Define execution configuration
    // int threadsPerBlock = ...;
    // int blocksPerGrid = ...;

    // TODO: Launch kernel
    // vectorAddKernel<<<...>>>(...);

    // Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}
