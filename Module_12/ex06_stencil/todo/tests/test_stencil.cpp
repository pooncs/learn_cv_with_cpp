#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "box_filter.h"

TEST(StencilTest, BlursImage) {
    int width = 32;
    int height = 32;
    size_t size = width * height;
    
    std::vector<unsigned char> h_in(size, 0);
    // Set central pixel to 255
    h_in[16 * width + 16] = 255;
    
    std::vector<unsigned char> h_out(size);
    
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    if (d_in == nullptr) return;

    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);
    
    boxFilterWrapper(d_in, d_out, width, height);
    
    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);
    
    // The central pixel (16,16) and its 8 neighbors should average to 255/9 = 28
    EXPECT_EQ(h_out[16 * width + 16], 28);
    EXPECT_EQ(h_out[16 * width + 15], 28);
    EXPECT_EQ(h_out[15 * width + 16], 28);
    
    // Far away pixel should be 0
    EXPECT_EQ(h_out[0], 0);
    
    cudaFree(d_in);
    cudaFree(d_out);
}
