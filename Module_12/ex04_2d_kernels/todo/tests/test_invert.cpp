#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "image_invert.h"

TEST(ImageInvertTest, InvertsColorsCorrectly) {
    int width = 32;
    int height = 32;
    int channels = 3;
    size_t size = width * height * channels;
    
    std::vector<unsigned char> h_in(size, 100);
    std::vector<unsigned char> h_out(size);
    
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    if (d_in == nullptr) return;

    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);
    
    invertImageWrapper(d_in, d_out, width, height, channels);
    
    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(h_out[i], 155) << "Mismatch at index " << i; // 255 - 100 = 155
    }
    
    cudaFree(d_in);
    cudaFree(d_out);
}
