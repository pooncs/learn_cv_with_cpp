#include "stream_add.h"
#include <vector>
#include <iostream>

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void streamAddWrapper(float* h_A, float* h_B, float* h_C, 
                      float* d_A, float* d_B, float* d_C, 
                      int N, int nStreams) 
{
    // Create streams
    std::vector<cudaStream_t> streams(nStreams);
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int chunkSize = (N + nStreams - 1) / nStreams;

    for (int i = 0; i < nStreams; ++i) {
        int offset = i * chunkSize;
        int currentSize = chunkSize;
        
        // Handle last chunk
        if (offset + currentSize > N) {
            currentSize = N - offset;
        }
        if (currentSize <= 0) break;

        size_t bytes = currentSize * sizeof(float);

        // 1. Copy H -> D (Async)
        cudaMemcpyAsync(&d_A[offset], &h_A[offset], bytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_B[offset], &h_B[offset], bytes, cudaMemcpyHostToDevice, streams[i]);

        // 2. Launch Kernel (Async)
        int blockSize = 256;
        int gridSize = (currentSize + blockSize - 1) / blockSize;
        
        // Note: Kernel takes absolute pointers, but uses offset to calculate index.
        // Alternatively, pass &d_A[offset] and reset index to 0. 
        // Let's pass full pointer + offset for clarity with index calculation.
        vectorAddKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_A, d_B, d_C, N, offset);

        // 3. Copy D -> H (Async)
        cudaMemcpyAsync(&h_C[offset], &d_C[offset], bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize everything
    cudaDeviceSynchronize();

    // Destroy streams
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}
