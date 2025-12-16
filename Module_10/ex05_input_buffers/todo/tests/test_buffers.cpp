#include <gtest/gtest.h>
#include <cuda_runtime.h>

TEST(CUDABuffers, Allocation) {
    void* d_ptr = nullptr;
    size_t size = 1024;
    
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err == cudaSuccess) {
        EXPECT_NE(d_ptr, nullptr);
        
        // Test Memset
        err = cudaMemset(d_ptr, 0, size);
        EXPECT_EQ(err, cudaSuccess);
        
        // Free
        err = cudaFree(d_ptr);
        EXPECT_EQ(err, cudaSuccess);
    } else {
        // If no GPU, skip test gracefully or warn
        std::cerr << "Skipping CUDA test: " << cudaGetErrorString(err) << std::endl;
        // In some CI envs without GPU, this is expected.
    }
}
