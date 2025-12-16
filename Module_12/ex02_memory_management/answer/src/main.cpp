#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
            return -1; \
        } \
    } while (0)

int main() {
    const size_t size = 100 * 1024 * 1024; // 100 MB
    const size_t bytes = size * sizeof(float);
    
    std::cout << "Benchmarking transfer of " << bytes / (1024*1024) << " MB..." << std::endl;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    // --- Pageable Memory ---
    float *h_pageable = new float[size];
    // Initialize to force physical allocation
    for(size_t i=0; i<size; i+=1024) h_pageable[i] = 1.0f;

    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    
    double pageable_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Pageable Transfer: " << pageable_ms << " ms" << std::endl;

    delete[] h_pageable;

    // --- Pinned Memory ---
    float *h_pinned;
    CHECK_CUDA(cudaMallocHost(&h_pinned, bytes));
    for(size_t i=0; i<size; i+=1024) h_pinned[i] = 1.0f;

    start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    end = std::chrono::high_resolution_clock::now();
    
    double pinned_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Pinned Transfer:   " << pinned_ms << " ms" << std::endl;

    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_data));

    if (pinned_ms < pageable_ms) {
        std::cout << "Success: Pinned is " << pageable_ms / pinned_ms << "x faster." << std::endl;
    } else {
        std::cout << "Note: Pinned was not faster. This can happen on small transfers or specific systems." << std::endl;
    }

    return 0;
}
