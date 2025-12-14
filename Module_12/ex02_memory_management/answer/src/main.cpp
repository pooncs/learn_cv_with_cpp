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

    // Initialize h_in
    std::iota(h_in.begin(), h_in.end(), 0); // 0, 1, 2, ...

    std::cout << "Allocating " << bytes << " bytes on GPU..." << std::endl;

    // 2. Allocate Device Memory
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    // 3. Copy Host -> Device
    std::cout << "Copying Host -> Device..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_data, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // 4. Copy Device -> Host (Round Trip)
    std::cout << "Copying Device -> Host..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    // 5. Verify
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != h_in[i]) {
            std::cerr << "Mismatch at index " << i << ": " << h_out[i] << " != " << h_in[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Round Trip Test: PASS" << std::endl;
    } else {
        std::cout << "Round Trip Test: FAIL" << std::endl;
    }

    // 6. Test Memset
    std::cout << "Testing cudaMemset (setting to 0)..." << std::endl;
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != 0) {
            std::cerr << "Mismatch at index " << i << ": " << h_out[i] << " != 0" << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Memset Test: PASS" << std::endl;
    } else {
        std::cout << "Memset Test: FAIL" << std::endl;
    }

    // 7. Free
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
