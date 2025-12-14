#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "vector_add.h"

TEST(VectorAddTest, AddsCorrectly) {
    int N = 1000;
    size_t bytes = N * sizeof(float);
    
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    if (d_A == nullptr) return; // Skip if no GPU
    
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    
    vectorAddWrapper(d_A, d_B, d_C, N);
    
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(h_C[i], 3.0f, 1e-5) << "Mismatch at index " << i;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
