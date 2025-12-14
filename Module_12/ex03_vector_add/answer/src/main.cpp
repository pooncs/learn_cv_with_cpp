#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "vector_add.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while (0)

int main() {
    int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);
    std::vector<float> h_Ref(N);

    // Initialize
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Ref[i] = h_A[i] + h_B[i];
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Launch
    std::cout << "Launching Vector Add for " << N << " elements..." << std::endl;
    vectorAddWrapper(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i) {
        maxError = std::max(maxError, std::abs(h_C[i] - h_Ref[i]));
    }

    std::cout << "Max Error: " << maxError << std::endl;
    if (maxError < 1e-5) {
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
