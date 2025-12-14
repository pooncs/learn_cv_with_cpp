#include <iostream>
#include <cuda_runtime.h>

// TODO: Define error checking macro if needed
// #define CUDA_CHECK(call) ...

int main() {
    int deviceCount = 0;
    // TODO: Get device count using cudaGetDeviceCount
    // ...

    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA" << std::endl;
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        // TODO: Set current device
        // ...
        
        cudaDeviceProp deviceProp;
        // TODO: Get device properties
        // ...

        std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"" << std::endl;
        
        // TODO: Print more properties
        // - Compute Capability
        // - Total Global Memory
        // - Shared Memory per Block
        // - Max Threads per Block
    }

    return 0;
}
