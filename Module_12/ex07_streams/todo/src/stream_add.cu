#include "stream_add.h"
#include <vector>
#include <iostream>

// TODO: Implement Kernel
// __global__ void vectorAddKernel(...)

void streamAddWrapper(float* h_A, float* h_B, float* h_C, 
                      float* d_A, float* d_B, float* d_C, 
                      int N, int nStreams) 
{
    // TODO: Create Streams
    // ...

    // TODO: Loop over streams and issue Async commands
    // cudaMemcpyAsync(H2D)
    // Kernel Launch
    // cudaMemcpyAsync(D2H)

    // TODO: Sync and Destroy Streams
}
