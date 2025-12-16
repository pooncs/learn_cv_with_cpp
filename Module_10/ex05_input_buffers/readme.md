# Exercise 05: Input Buffers

## Goal
Allocate and manage GPU memory for TensorRT inference. Understand the difference between Host (CPU) and Device (GPU) memory.

## Learning Objectives
1.  Use `cudaMalloc` to allocate device memory.
2.  Use `cudaMemcpy` to transfer data between Host and Device.
3.  Understand TensorRT bindings (input/output pointers).

## Practical Motivation
TensorRT execution runs on the GPU. It cannot access your CPU variables directly. You must move data to the GPU before inference and move results back after.

## Theory: Memory Management
*   **Host Memory:** Standard RAM (access via `malloc`, `new`, `std::vector`).
*   **Device Memory:** GPU VRAM (access via `cudaMalloc`).
*   **Transfer:** `cudaMemcpy(dst, src, size, kind)`.
    *   `cudaMemcpyHostToDevice`
    *   `cudaMemcpyDeviceToHost`

## Step-by-Step Instructions

### Task 1: Allocate Device Memory
Open `todo/src/main.cpp`.
1.  Define input size (e.g., 1 * 3 * 224 * 224 * sizeof(float)).
2.  Call `cudaMalloc(&d_input, size)`. Check for errors.
3.  Allocate output memory similarly.

### Task 2: Host Data Preparation
1.  Create a `std::vector<float>` with dummy data.

### Task 3: Transfer H2D
1.  Call `cudaMemcpy` to copy vector data to `d_input`.

### Task 4: Transfer D2H (Simulation)
1.  (After hypothetical inference) Copy `d_output` back to a host vector.

### Task 5: Cleanup
1.  `cudaFree(d_input)`.
2.  `cudaFree(d_output)`.

## Verification
Run the app. Ensure no CUDA errors are printed.
