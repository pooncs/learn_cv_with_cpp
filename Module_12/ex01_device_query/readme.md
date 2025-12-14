# Exercise 01: CUDA Device Query

## Goal
Understand how to interact with the CUDA Runtime API to query the properties of the available GPU(s). This is the "Hello World" of understanding your hardware capabilities before launching kernels.

## Learning Objectives
1.  Initialize the CUDA runtime (implicitly).
2.  Count the number of available CUDA-capable devices.
3.  Retrieve and interpret `cudaDeviceProp` struct.
4.  Understand key hardware limits: Max threads per block, Shared memory size, Compute Capability.

## Practical Motivation
Before deploying a high-performance CV algorithm, your application must know:
-   **Is a GPU available?** If not, fallback to CPU.
-   **How much memory is available?** To avoid OOM (Out of Memory) errors.
-   **What is the Compute Capability?** To determine which features (e.g., Tensor Cores, FP16) are supported.
-   **Max Threads per Block:** To configure kernel launch parameters (`<<<grid, block>>>`) correctly.

## Theory: The CUDA Device API
The CUDA Runtime API provides functions to manage devices.
-   `cudaGetDeviceCount(int* count)`: Returns the number of devices.
-   `cudaGetDeviceProperties(cudaDeviceProp* prop, int device)`: Fills a struct with device details.

Key properties in `cudaDeviceProp`:
-   `name`: ASCII string identifying the device (e.g., "NVIDIA GeForce RTX 3090").
-   `totalGlobalMem`: Global memory available on device in bytes.
-   `sharedMemPerBlock`: Maximum shared memory available per block in bytes.
-   `warpSize`: Warp size in threads (usually 32).
-   `maxThreadsPerBlock`: Maximum number of threads per block.
-   `major`, `minor`: Compute capability version.

## Step-by-Step Instructions

### Task 1: Check for CUDA Devices
Open `src/main.cpp`.
1.  Use `cudaGetDeviceCount` to get the number of devices.
2.  If the count is 0, print a message and exit.
3.  Handle potential errors using a helper macro or function (check `cudaGetErrorString`).

### Task 2: Print Device Properties
For each device found:
1.  Call `cudaGetDeviceProperties`.
2.  Print the following details formatted nicely:
    -   Device Name
    -   Compute Capability (Major.Minor)
    -   Total Global Memory (in MB or GB)
    -   Shared Memory per Block (in KB)
    -   Max Threads per Block
    -   Multi-Processor Count

### Task 3: Calculate Theoretical Peak Bandwidth (Optional but Recommended)
Using `memoryClockRate` and `memoryBusWidth`:
$$ Bandwidth (GB/s) = \frac{Memory Clock (kHz) \times Bus Width (bits) \times 2 (DDR)}{8 \times 10^6} $$

## Common Pitfalls
-   **No Driver Installed:** `cudaGetDeviceCount` might return an error if the NVIDIA driver is missing or version mismatch.
-   **Incorrect Units:** Memory is returned in bytes. Be careful when converting to KB/MB/GB (divide by 1024 vs 1000). `clockRate` is in kilohertz.

## Code Hints
```cpp
// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err); \
        } \
    } while (0)

int deviceCount;
CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
```

## Verification
Run the program. You should see output similar to:
```text
Device 0: NVIDIA GeForce RTX 3080
  Compute Capability: 8.6
  Total Global Mem: 10240 MB
  ...
```
