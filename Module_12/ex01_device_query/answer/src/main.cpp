#include <iostream>
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
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << "\n-> " << cudaGetErrorString(error_id) << std::endl;
        std::cerr << "Result = FAIL" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA" << std::endl;
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

        std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"" << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << (float)deviceProp.totalGlobalMem / 1048576.0f << " MBytes (" << deviceProp.totalGlobalMem << " bytes)" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024.0f << " KBytes" << std::endl;
        std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per MultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  MultiProcessor Count: " << deviceProp.multiProcessorCount << std::endl;
        
        // Theoretical bandwidth
        double memoryClock = deviceProp.memoryClockRate; // in kHz
        double memBusWidth = deviceProp.memoryBusWidth;
        double peakBandwidth = 2.0 * memoryClock * (memBusWidth / 8.0) / 1.0e6;
        std::cout << "  Memory Clock Rate: " << memoryClock / 1000.0 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << memBusWidth << "-bit" << std::endl;
        std::cout << "  Peak Memory Bandwidth: " << peakBandwidth << " GB/s" << std::endl;
    }

    std::cout << "\nResult = PASS" << std::endl;
    return 0;
}
