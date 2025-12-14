#include <gtest/gtest.h>
#include <cuda_runtime.h>

TEST(DeviceQueryTest, CanGetDeviceCount) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err == cudaSuccess) {
        if (deviceCount > 0) {
            SUCCEED() << "Found " << deviceCount << " CUDA devices.";
        } else {
            SUCCEED() << "No CUDA devices found (count is 0), but API call succeeded.";
        }
    } else {
        // Allow failure if no driver or device, but print why
        FAIL() << "cudaGetDeviceCount failed: " << cudaGetErrorString(err);
    }
}

TEST(DeviceQueryTest, CanGetDeviceProperties) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, 0);
        EXPECT_EQ(err, cudaSuccess) << "Failed to get properties for device 0";
    }
}
