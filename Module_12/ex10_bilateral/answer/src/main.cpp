#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include "bilateral.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while (0)

void writePPM(const char* filename, const unsigned char* data, int w, int h) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P5\n" << w << " " << h << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(data), w * h);
}

int main() {
    int width = 512;
    int height = 512;
    size_t bytes = width * height * sizeof(unsigned char);

    // Create synthetic noisy image with edge
    std::vector<unsigned char> h_in(bytes);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Step edge at x=256
            int val = (x < 256) ? 50 : 200;
            // Add noise
            int noise = (rand() % 20) - 10;
            val = std::max(0, std::min(255, val + noise));
            h_in[y * width + x] = (unsigned char)val;
        }
    }

    unsigned char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    float sigma_s = 2.0f; // Spatial sigma
    float sigma_r = 30.0f; // Range sigma (intensity)

    std::cout << "Launching Bilateral Filter..." << std::endl;
    bilateralFilterWrapper(d_in, d_out, width, height, sigma_s, sigma_r);

    std::vector<unsigned char> h_out(bytes);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify
    // 1. Noise reduction in flat region
    // 2. Edge preservation at x=256
    
    // Check flat region x=100
    int sumFlat = 0;
    for(int y=100; y<200; ++y) sumFlat += h_out[y*width + 100];
    double avgFlat = sumFlat / 100.0;
    
    // Check edge region
    int leftVal = h_out[256*width + 254];
    int rightVal = h_out[256*width + 258];
    
    std::cout << "Avg Flat Region (orig ~50): " << avgFlat << std::endl;
    std::cout << "Edge Left (orig ~50): " << leftVal << std::endl;
    std::cout << "Edge Right (orig ~200): " << rightVal << std::endl;

    if (abs(avgFlat - 50.0) < 5.0 && rightVal - leftVal > 100) {
        std::cout << "Result: PASS" << std::endl;
        writePPM("bilateral_out.pgm", h_out.data(), width, height);
    } else {
        std::cout << "Result: FAIL (Check thresholds)" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
