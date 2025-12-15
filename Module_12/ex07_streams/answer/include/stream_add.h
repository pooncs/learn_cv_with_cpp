#pragma once
#include <cuda_runtime.h>

void streamAddWrapper(float* h_A, float* h_B, float* h_C, 
                      float* d_A, float* d_B, float* d_C, 
                      int N, int nStreams);
