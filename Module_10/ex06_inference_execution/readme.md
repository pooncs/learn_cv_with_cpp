# Exercise 06: Inference Execution

## Goal
Run inference using the TensorRT `IExecutionContext`. This involves copying data to the GPU, executing the model, and copying results back to the CPU.

## Learning Objectives
1.  **Memory Management:** Allocate GPU memory (`cudaMalloc`) for inputs and outputs.
2.  **Data Transfer:** Use `cudaMemcpy` to move data Host <-> Device.
3.  **Binding Indices:** Retrieve input/output binding indices from the engine.
4.  **Execution:** Call `context->executeV2(bindings)` (or `enqueueV2` for async).

## Practical Motivation
The engine is ready (Ex04), and the buffers are allocated (Ex05). Now we need to actually run the data through the model.

**Analogy:**
*   **Ex05 (Buffers):** Setting the table with empty plates (GPU memory).
*   **Ex06 (Inference):** 
    1.  Putting food on the plates (Host -> Device copy).
    2.  Eating the meal (Inference execution).
    3.  Taking the clean/dirty plates back to the kitchen (Device -> Host copy).

## Theory: Synchronous vs Asynchronous
*   **Synchronous (`executeV2`):** CPU waits for GPU to finish. Simple but blocks the CPU.
*   **Asynchronous (`enqueueV2`):** CPU schedules work and returns immediately. Allows CPU to do other things (like pre-processing the next frame) while GPU works. Requires `cudaStream_t`.

## Step-by-Step Instructions

### Task 1: Setup
1.  Load engine (reuse code from Ex04).
2.  Allocate GPU buffers (reuse code from Ex05 or do it simply here).
3.  Create CUDA stream (optional but recommended).

### Task 2: Prepare Input
1.  Create dummy input data on Host (CPU).
2.  `cudaMemcpyAsync` Host -> Device.

### Task 3: Execute
1.  Create an array of pointers `void* bindings[]` pointing to input and output GPU buffers.
2.  Call `context->enqueueV2(bindings, stream, nullptr)`.

### Task 4: Retrieve Output
1.  `cudaMemcpyAsync` Device -> Host.
2.  `cudaStreamSynchronize(stream)` to wait for everything to finish.

## Code Hints
```cpp
// Get indices
int inputIndex = engine->getBindingIndex("input");
int outputIndex = engine->getBindingIndex("output");

void* buffers[2];
buffers[inputIndex] = inputBuffer; // GPU ptr
buffers[outputIndex] = outputBuffer; // GPU ptr

// H2D
cudaMemcpyAsync(inputBuffer, hostInput.data(), size, cudaMemcpyHostToDevice, stream);

// Run
context->enqueueV2(buffers, stream, nullptr);

// D2H
cudaMemcpyAsync(hostOutput.data(), outputBuffer, size, cudaMemcpyDeviceToHost, stream);

// Wait
cudaStreamSynchronize(stream);
```

## Verification
Print the first few values of the output buffer. They should be non-zero (if the model does something).
