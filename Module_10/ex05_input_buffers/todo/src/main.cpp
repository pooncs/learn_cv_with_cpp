#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            return -1; \
        } \
    } while (0)

int main() {
    // Input Dimensions: 1x3x224x224
    size_t input_elements = 1 * 3 * 224 * 224;
    size_t input_size = input_elements * sizeof(float);
    
    // Output Dimensions: 1x1000 (ImageNet classes)
    size_t output_elements = 1 * 1000;
    size_t output_size = output_elements * sizeof(float);

    // TODO: Allocate Device Memory (d_input, d_output)
    
    // TODO: Prepare Host Data (h_input)
    
    // TODO: Copy Host -> Device
    
    std::cout << "Data transferred to GPU." << std::endl;

    // TODO: Copy Device -> Host (h_output)
    
    // TODO: Free Memory
    
    return 0;
}
