# Exercise 06: Inference Execution

## Goal
Perform synchronous inference using a TensorRT execution context.

## Learning Objectives
1.  Create `IExecutionContext` from `ICudaEngine`.
2.  Set input binding dimensions (if dynamic).
3.  Run inference using `enqueueV2` (async) or `executeV2` (sync).
4.  Synchronize using `cudaStream`.

## Practical Motivation
This is the core runtime loop. Once you have an engine and buffers, you call `execute` repeatedly for every frame.

## Prerequisites
*   `simple_model.engine` (from Ex 03/04).
*   GPU access.

## Step-by-Step Instructions

### Task 1: Load Engine
Open `todo/src/main.cpp`.
1.  Deserialize the engine (reuse code from Ex 04 or use helper).

### Task 2: Create Context
1.  `auto context = engine->createExecutionContext();`.

### Task 3: Setup Buffers
1.  Allocate CUDA buffers (Ex 05).
2.  Create an array `void* bindings[]` containing `{d_input, d_output}`. Order must match `engine->getBindingIndex(name)`.

### Task 4: Execute
1.  Copy H2D.
2.  `context->executeV2(bindings)`.
3.  Copy D2H.

## Verification
Run inference. Print output values.
