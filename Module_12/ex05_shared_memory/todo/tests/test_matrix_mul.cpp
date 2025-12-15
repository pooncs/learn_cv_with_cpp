#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "matrix_mul.h"

TEST(MatrixMulTest, MultipliesCorrectly) {
    int width = 32; // Small multiple of 16
    size_t size = width * width * sizeof(float);
    
    std::vector<float> h_A(width * width, 1.0f); // Identity-like check
    std::vector<float> h_B(width * width, 2.0f);
    std::vector<float> h_C(width * width);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    if (d_A == nullptr) return;

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);
    
    matrixMulWrapper(d_A, d_B, d_C, width);
    
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    // Result should be 1.0 * 2.0 * width = 2.0 * 32 = 64.0
    for (size_t i = 0; i < width * width; ++i) {
        EXPECT_NEAR(h_C[i], 64.0f, 1e-3);
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
