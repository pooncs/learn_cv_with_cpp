# Exercise 03: TensorRT Builder

## Goal
Convert an ONNX model into a highly optimized TensorRT Engine (`.plan` file).

## Learning Objectives
1.  Initialize `nvinfer1::IBuilder`.
2.  Parse an ONNX model using `nvonnxparser::IParser`.
3.  Configure builder config (FP16/FP32 modes, max workspace).
4.  Build and serialize the engine.

## Practical Motivation
ONNX Runtime is fast, but TensorRT is faster on NVIDIA GPUs because it performs layer fusion, kernel auto-tuning, and precision calibration specific to the target hardware.

## Prerequisites
*   NVIDIA GPU + Drivers.
*   CUDA Toolkit installed.
*   TensorRT installed.
*   `simple_model.onnx` from Exercise 01.

## Theory: The Builder
The `IBuilder` takes a `INetworkDefinition` (populated by the ONNX Parser) and an `IBuilderConfig` to generate an `ICudaEngine`. This process is time-consuming, so we usually save (serialize) the result to disk.

## Step-by-Step Instructions

### Task 1: Initialize Builder & Network
Open `todo/src/main.cpp`.
1.  Create `Logger` implementing `nvinfer1::ILogger`.
2.  Create `IBuilder`.
3.  Create `INetworkDefinition` (with `1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)`).
4.  Create `IParser`.

### Task 2: Parse ONNX
1.  Load `simple_model.onnx`.
2.  `parser->parseFromFile(...)`. Check for errors.

### Task 3: Build Engine
1.  Create `IBuilderConfig`.
2.  Set memory pool limit (workspace size).
3.  (Optional) Set FP16 flag if supported.
4.  `builder->buildSerializedNetwork(...)`.

### Task 4: Save to File
1.  Write the resulting `IHostMemory` blob to `simple_model.engine`.

## Verification
Run the builder. It should produce `simple_model.engine`.
