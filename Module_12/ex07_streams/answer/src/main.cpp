#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "stream_add.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while (0)

int main() {
    int N = 1 << 22; // 4M elements
    size_t bytes = N * sizeof(float);
    int nStreams = 4;

    std::cout << "Vector Addition with " << nStreams << " streams for " << N << " elements." << std::endl;

    // Use Pinned Memory for Async Transfers
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost((void**)&h_A, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, bytes));

    // Initialize
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device Memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Run Wrapper
    streamAddWrapper(h_A, h_B, h_C, d_A, d_B, d_C, N, nStreams);

    // Verify
    bool pass = true;
    for (int i = 0; i < N; ++i) {
        if (abs(h_C[i] - 3.0f) > 1e-5) {
            pass = false;
            std::cerr << "Mismatch at " << i << ": " << h_C[i] << std::endl;
            break;
        }
    }

    if (pass) std::cout << "Result: PASS" << std::endl;
    else std::cout << "Result: FAIL" << std::endl;

    // Cleanup
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
