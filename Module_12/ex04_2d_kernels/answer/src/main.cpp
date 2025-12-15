#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include "image_invert.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while (0)

// Minimal PPM Writer for testing without OpenCV
void writePPM(const char* filename, const unsigned char* data, int w, int h) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << w << " " << h << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(data), w * h * 3);
}

int main(int argc, char** argv) {
    int width = 512;
    int height = 512;
    int channels = 3;
    size_t bytes = width * height * channels * sizeof(unsigned char);

    std::cout << "Creating synthetic image " << width << "x" << height << "..." << std::endl;
    std::vector<unsigned char> h_in(bytes);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            h_in[idx + 0] = x % 255;     // R
            h_in[idx + 1] = y % 255;     // G
            h_in[idx + 2] = (x + y) % 255; // B
        }
    }

    unsigned char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    std::cout << "Launching Kernel..." << std::endl;
    invertImageWrapper(d_in, d_out, width, height, channels);

    std::vector<unsigned char> h_out(bytes);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify
    bool pass = true;
    for (int i = 0; i < bytes; ++i) {
        if (h_out[i] != (255 - h_in[i])) {
            pass = false;
            if (i < 10) std::cerr << "Mismatch at " << i << ": " << (int)h_out[i] << " != " << (255 - (int)h_in[i]) << std::endl;
        }
    }

    if (pass) {
        std::cout << "Result: PASS" << std::endl;
        writePPM("inverted_output.ppm", h_out.data(), width, height);
        std::cout << "Saved inverted_output.ppm" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
