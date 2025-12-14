#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while (0)

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(int);

    // 1. Allocate Host Memory
    std::vector<int> h_in(N);
    std::vector<int> h_out(N);
    std::iota(h_in.begin(), h_in.end(), 0);

    // 2. Allocate Device Memory
    int* d_data = nullptr;
    // TODO: Allocate memory on device using cudaMalloc
    // ...

    // 3. Copy Host -> Device
    std::cout << "Copying Host -> Device..." << std::endl;
    // TODO: Copy data from h_in to d_data using cudaMemcpy
    // ...

    // 4. Copy Device -> Host (Round Trip)
    std::cout << "Copying Device -> Host..." << std::endl;
    // TODO: Copy data from d_data to h_out using cudaMemcpy
    // ...

    // 5. Verify
    // ...

    // 6. Test Memset
    std::cout << "Testing cudaMemset (setting to 0)..." << std::endl;
    // TODO: Set d_data to 0 using cudaMemset
    // ...
    // TODO: Copy back to h_out
    // ...

    // 7. Free
    // TODO: Free device memory
    // ...

    return 0;
}
