#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "bilateral.h"

TEST(BilateralTest, RunsWithoutError) {
    int width = 32;
    int height = 32;
    size_t size = width * height;
    
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    cudaMemset(d_in, 128, size); // Uniform gray
    
    if (d_in == nullptr) return;

    bilateralFilterWrapper(d_in, d_out, width, height, 2.0f, 10.0f);
    
    std::vector<unsigned char> h_out(size);
    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);
    
    // Uniform input should result in uniform output (roughly)
    EXPECT_NEAR(h_out[0], 128, 1);
    
    cudaFree(d_in);
    cudaFree(d_out);
}
