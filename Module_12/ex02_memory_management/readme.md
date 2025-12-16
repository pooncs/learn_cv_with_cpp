# Exercise 02: Pinned vs Pageable Memory

## Goal
Measure the performance difference between standard `malloc` (Pageable) and `cudaMallocHost` (Pinned) memory when transferring data to the GPU.

## Learning Objectives
1.  **Pageable Memory:** Default system memory, can be swapped out to disk.
2.  **Pinned Memory:** Locked in RAM, allows DMA (Direct Memory Access) for faster transfers.
3.  **cudaHostAlloc / cudaMallocHost:** API to allocate pinned memory.
4.  **Performance Benchmarking:** Using `std::chrono` to measure transfer time.

## Practical Motivation
In High-Frequency Trading or Real-Time Video Processing (60 FPS+), every millisecond counts. Copying data from CPU to GPU is often the bottleneck. Pinned memory can speed this up by 2x-3x.

**Analogy:**
*   **Pageable Memory:** A passenger checking in luggage. The agent (OS) might move the bag to a back room (Swap) temporarily. When the plane (GPU) needs it, the agent has to go find it and bring it back to the counter before loading.
*   **Pinned Memory:** Carry-on luggage. It stays with you (RAM) at all times and goes directly onto the plane (GPU) without intermediate checks.

## Theory: DMA
DMA (Direct Memory Access) allows the GPU to read system RAM without involving the CPU. However, the OS must guarantee the data stays at the same physical address (Pinned). If it's pageable, the OS might move it, breaking the DMA transfer.

## Step-by-Step Instructions

### Task 1: Pageable Transfer
1.  Allocate 100MB using `new` or `malloc`.
2.  Measure time to `cudaMemcpy` to device.
3.  Free memory.

### Task 2: Pinned Transfer
1.  Allocate 100MB using `cudaMallocHost`.
2.  Measure time to `cudaMemcpy` to device.
3.  Free using `cudaFreeHost`.

### Task 3: Compare
1.  Print the speedup factor.

## Code Hints
```cpp
// Pinned Allocation
float* h_pinned;
cudaMallocHost(&h_pinned, size_bytes);

// Timing
auto start = std::chrono::high_resolution_clock::now();
cudaMemcpy(d_ptr, h_pinned, size_bytes, cudaMemcpyHostToDevice);
auto end = std::chrono::high_resolution_clock::now();
```

## Verification
Pinned memory transfer should be significantly faster (lower duration) than Pageable.
