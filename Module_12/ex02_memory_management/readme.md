# Exercise 02: CUDA Memory Management

## Goal
Master the fundamental memory management functions in CUDA: `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.

## Learning Objectives
1.  Understand the difference between Host (CPU) and Device (GPU) memory spaces.
2.  Allocate memory on the GPU using `cudaMalloc`.
3.  Transfer data between Host and Device using `cudaMemcpy`.
4.  Free GPU memory to prevent leaks using `cudaFree`.
5.  Initialize GPU memory using `cudaMemset`.

## Practical Motivation
In a typical heterogeneous computing pipeline (CPU + GPU), data originates on the CPU (e.g., loading an image from disk). To process it on the GPU, you must:
1.  Allocate space on the GPU.
2.  Copy the data from CPU to GPU.
3.  Launch kernels (covered in next exercises).
4.  Copy results back to CPU.
5.  Clean up.

Failure to manage memory correctly leads to segmentation faults (accessing device pointer on host) or Out-Of-Memory (OOM) errors.

## Theory: Host vs Device Pointers
-   **Host Pointer:** Points to RAM. Accessible by CPU.
-   **Device Pointer:** Points to VRAM. Accessible by GPU.
-   **Dereferencing:**
    -   `*h_ptr` on CPU: OK.
    -   `*d_ptr` on CPU: **CRASH** (Segfault).
    -   `*d_ptr` on GPU: OK.

## Step-by-Step Instructions

### Task 1: Allocation
1.  Define a vector size $N$ (e.g., 1024).
2.  Allocate two host arrays `h_in` and `h_out` of size $N$ (float or int).
3.  Initialize `h_in` with values (e.g., $i$ for index $i$).
4.  Allocate a device array `d_data` of size $N$ using `cudaMalloc`.
    -   **Signature:** `cudaError_t cudaMalloc(void** devPtr, size_t size);`
    -   **Usage:** `float* d_data; cudaMalloc((void**)&d_data, N * sizeof(float));`

### Task 2: Host to Device Transfer
1.  Copy `h_in` to `d_data` using `cudaMemcpy`.
    -   **Signature:** `cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);`
    -   **Kind:** `cudaMemcpyHostToDevice`.

### Task 3: Device Modification (Memset)
Since we haven't written kernels yet, use `cudaMemset` to set the values to 0 or another byte value, OR simply copy it back to verify the round trip.
Let's try:
1.  Copy `h_in` to `d_data`.
2.  Copy `d_data` back to `h_out`.
3.  Verify `h_out` == `h_in`.

**Challenge:** Try `cudaMemset(d_data, 0, size)` in between and verify `h_out` is all zeros.

### Task 4: Cleanup
1.  Free `d_data` using `cudaFree`.
2.  Free host memory (if using `malloc`/`new`).

## Common Pitfalls
-   **Size in Bytes:** `cudaMalloc` and `cudaMemcpy` take size in **bytes**, not number of elements. Always multiply by `sizeof(T)`.
-   **Wrong Direction:** `cudaMemcpy(dest, src, ...)` follows C-style `memcpy`. Using `HostToDevice` when copying `d` to `h` will corrupt memory or crash.
-   **Double Free:** Freeing the same pointer twice.

## Code Hints
```cpp
int N = 1024;
size_t bytes = N * sizeof(int);

int *h_in = (int*)malloc(bytes);
int *d_data = nullptr;

// Allocate Device
cudaMalloc((void**)&d_data, bytes);

// Copy H -> D
cudaMemcpy(d_data, h_in, bytes, cudaMemcpyHostToDevice);

// Free
cudaFree(d_data);
free(h_in);
```

## Verification
The program should print "PASS" if the data copied back matches the input (or the memset result).
