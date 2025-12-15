#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "stream_add.h"

TEST(StreamTest, AddsCorrectlyWithStreams) {
    int N = 1000;
    size_t bytes = N * sizeof(float);
    int nStreams = 2;
    
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, bytes);
    cudaMallocHost(&h_B, bytes);
    cudaMallocHost(&h_C, bytes);
    
    for(int i=0; i<N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    streamAddWrapper(h_A, h_B, h_C, d_A, d_B, d_C, N, nStreams);
    
    for(int i=0; i<N; ++i) {
        EXPECT_NEAR(h_C[i], 3.0f, 1e-5);
    }
    
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
