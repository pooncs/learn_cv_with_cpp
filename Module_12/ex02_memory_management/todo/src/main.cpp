#include <iostream>
#include <vector>
#include <chrono>
// #include <cuda_runtime.h>

int main() {
    std::cout << "Pinned Memory Benchmark" << std::endl;

    // TODO: Allocate Device Memory

    // TODO: 1. Pageable Memory Benchmark
    // Allocate new float[]
    // Measure cudaMemcpy
    // Free

    // TODO: 2. Pinned Memory Benchmark
    // Allocate cudaMallocHost
    // Measure cudaMemcpy
    // Free cudaFreeHost

    // TODO: Compare Results

    return 0;
}
