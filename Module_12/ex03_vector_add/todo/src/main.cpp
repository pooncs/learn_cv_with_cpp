#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "vector_add.h"

// TODO: Error checking macro

int main() {
    int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // TODO: Allocate and initialize host vectors
    // ...

    // TODO: Allocate device memory
    // ...

    // TODO: Copy Host -> Device
    // ...

    // Launch
    std::cout << "Launching Vector Add for " << N << " elements..." << std::endl;
    // vectorAddWrapper(d_A, d_B, d_C, N);

    // TODO: Copy Device -> Host
    // ...

    // TODO: Verify
    // ...

    // TODO: Free memory
    // ...

    return 0;
}
