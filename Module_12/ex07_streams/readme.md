# Exercise 07: CUDA Streams

## Goal
Learn how to use CUDA Streams to overlap memory transfers (Host-to-Device and Device-to-Host) with kernel execution, increasing overall throughput.

## Learning Objectives
1.  Understand the concept of the "Default Stream" vs. "Non-Default Streams".
2.  Use `cudaStreamCreate` and `cudaStreamDestroy`.
3.  Use asynchronous memory copies: `cudaMemcpyAsync`.
4.  Launch kernels into specific streams.
5.  Pin host memory using `cudaHostAlloc` (Pinned Memory) which is required for overlap.

## Practical Motivation
By default, CUDA operations in the same stream are serialized. However, the GPU has separate engines for Copy and Compute. To use them simultaneously, we must issue commands in independent streams.
Scenario: Processing a video.
-   Stream 1: Copy Frame N to GPU.
-   Stream 2: Process Frame N-1.
-   Stream 3: Copy Frame N-2 back to CPU.

## Theory: Pinned Memory
Standard `malloc` allocates pageable memory. The OS can move this memory or swap it out. The GPU DMA engine cannot access pageable memory safely/efficiently.
-   **Pageable:** Driver must copy data to a temporary pinned buffer -> DMA to GPU. (Slow, Synchronous).
-   **Pinned (`cudaHostAlloc`):** Memory is locked in RAM. DMA can access directly. (Fast, Asynchronous).

## Step-by-Step Instructions

### Task 1: The Kernel (`src/stream_add.cu`)
Reuse the vector addition kernel from Exercise 03. It's simple enough to show overlap effects.

### Task 2: Host Code
1.  Allocate Host Memory using `cudaHostAlloc` (or `cudaMallocHost`).
2.  Create $N$ streams (e.g., 4).
3.  Divide the data into $N$ chunks.
4.  Loop over chunks:
    -   `cudaMemcpyAsync(..., stream[i])` (H2D)
    -   `vectorAddKernel<<<..., stream[i]>>>`
    -   `cudaMemcpyAsync(..., stream[i])` (D2H)
5.  `cudaDeviceSynchronize()`.
6.  Cleanup: Destroy streams and free pinned memory (`cudaFreeHost`).

## Common Pitfalls
-   **Not using Pinned Memory:** `cudaMemcpyAsync` will degrade to synchronous behavior if host pointer is not pinned.
-   **Default Stream:** If you forget the stream argument in kernel launch `<<<grid, block, 0, stream>>>`, it goes to stream 0, which serializes everything.

## Code Hints
```cpp
cudaStream_t streams[4];
for(int i=0; i<4; ++i) cudaStreamCreate(&streams[i]);

// Loop
for(int i=0; i<4; ++i) {
    int offset = i * chunkSize;
    cudaMemcpyAsync(&d_A[offset], &h_A[offset], size, H2D, streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(...);
    cudaMemcpyAsync(&h_C[offset], &d_C[offset], size, D2H, streams[i]);
}
```

## Verification
The program should produce correct addition results. Ideally, measuring time would show speedup compared to synchronous version, but for small data, overhead might hide it. The correctness check is primary here.
