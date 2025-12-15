#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "matrix_mul.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while (0)

int main() {
    int width = 512; // Must be multiple of 16 for simple kernel
    size_t bytes = width * width * sizeof(float);

    std::vector<float> h_A(width * width);
    std::vector<float> h_B(width * width);
    std::vector<float> h_C(width * width);
    std::vector<float> h_Ref(width * width);

    // Initialize
    for (int i = 0; i < width * width; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    std::cout << "Launching Matrix Mul " << width << "x" << width << "..." << std::endl;
    matrixMulWrapper(d_A, d_B, d_C, width);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify CPU
    // Only check a subset for speed, or small size
    std::cout << "Verifying..." << std::endl;
    float maxError = 0.0f;
    for (int y = 0; y < width; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += h_A[y * width + k] * h_B[k * width + x];
            }
            float diff = std::abs(h_C[y * width + x] - sum);
            if (diff > maxError) maxError = diff;
        }
    }

    std::cout << "Max Error: " << maxError << std::endl;
    if (maxError < 1e-3) { // Floating point arithmetic accumulation
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
