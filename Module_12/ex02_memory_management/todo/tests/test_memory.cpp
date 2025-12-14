#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>

TEST(MemoryManagementTest, AllocationAndFree) {
    int* d_ptr = nullptr;
    size_t size = 1024 * sizeof(int);
    
    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    if (err != cudaSuccess) {
        // Skip if no device
        if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
            SUCCEED() << "No CUDA device/driver available.";
            return;
        }
        FAIL() << "cudaMalloc failed: " << cudaGetErrorString(err);
    }
    
    EXPECT_NE(d_ptr, nullptr);
    
    err = cudaFree(d_ptr);
    EXPECT_EQ(err, cudaSuccess) << "cudaFree failed: " << cudaGetErrorString(err);
}

TEST(MemoryManagementTest, HostDeviceTransfer) {
    const int N = 100;
    size_t bytes = N * sizeof(int);
    
    std::vector<int> h_in(N);
    std::iota(h_in.begin(), h_in.end(), 0);
    
    int* d_ptr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_ptr, bytes);
    
    if (err != cudaSuccess) return; // Skip
    
    // H -> D
    err = cudaMemcpy(d_ptr, h_in.data(), bytes, cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess) << "H->D copy failed";
    
    // D -> H
    std::vector<int> h_out(N);
    err = cudaMemcpy(h_out.data(), d_ptr, bytes, cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess) << "D->H copy failed";
    
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(h_in[i], h_out[i]) << "Data mismatch at index " << i;
    }
    
    cudaFree(d_ptr);
}

TEST(MemoryManagementTest, Memset) {
    const int N = 100;
    size_t bytes = N * sizeof(int);
    
    int* d_ptr = nullptr;
    cudaMalloc((void**)&d_ptr, bytes);
    if (d_ptr == nullptr) return;
    
    cudaMemset(d_ptr, 0, bytes);
    
    std::vector<int> h_out(N);
    cudaMemcpy(h_out.data(), d_ptr, bytes, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(h_out[i], 0) << "Memset failed at index " << i;
    }
    
    cudaFree(d_ptr);
}
