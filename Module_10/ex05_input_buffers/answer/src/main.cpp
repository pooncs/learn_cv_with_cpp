#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <numeric>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            return -1; \
        } \
    } while (0)

int main() {
    // 1. Define Sizes
    const int batch = 1;
    const int channels = 3;
    const int height = 224;
    const int width = 224;
    
    size_t input_elements = batch * channels * height * width;
    size_t input_bytes = input_elements * sizeof(float);
    
    size_t output_elements = batch * 1000;
    size_t output_bytes = output_elements * sizeof(float);

    // 2. Allocate Device Memory
    void* d_input = nullptr;
    void* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    // 3. Prepare Host Data
    std::vector<float> h_input(input_elements);
    // Fill with dummy data
    std::fill(h_input.begin(), h_input.end(), 1.0f);

    std::vector<float> h_output(output_elements);

    // 4. Transfer Host -> Device
    std::cout << "Copying " << input_bytes << " bytes to GPU..." << std::endl;
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    // 5. Simulate Inference (Device -> Device)
    // In real scenario, TRT execute() happens here.
    // For now, let's just memset output to 0 or copy input to output (if sizes matched, but they don't)
    CHECK_CUDA(cudaMemset(d_output, 0, output_bytes));

    // 6. Transfer Device -> Host
    std::cout << "Copying results back to CPU..." << std::endl;
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    // 7. Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    std::cout << "Success!" << std::endl;
    return 0;
}
