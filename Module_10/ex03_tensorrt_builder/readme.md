# Exercise 03: TensorRT Builder

## Goal
Convert an ONNX model into an optimized TensorRT Engine (.plan) file.

## Learning Objectives
1.  **TensorRT API:** Understand `IBuilder`, `INetworkDefinition`, and `IBuilderConfig`.
2.  **ONNX Parser:** Use `nvonnxparser::IParser` to import the ONNX graph.
3.  **Optimization Profiles:** Set up dynamic shapes (optional but good practice).
4.  **Serialization:** Save the optimized engine to disk.

## Practical Motivation
ONNX is a format for *interchange*. It's not optimized for your specific GPU. TensorRT compiles the graph, fuses layers, selects the best kernels (fp32, fp16, int8) for your specific hardware, often resulting in 2x-10x speedups.

**Analogy:** ONNX is like a blueprint for a car engine drawn by an architect. It describes how it works but isn't a physical engine. TensorRT is the factory that takes the blueprint, optimizes the manufacturing process for its specific machines (your GPU), and builds the actual engine (Plan file) that runs as fast as possible.

## Theory: The Build Process
1.  **Builder:** The factory manager.
2.  **Network:** The empty chassis.
3.  **Parser:** Reads the blueprint (ONNX) and fills the chassis (Network).
4.  **Config:** Specifications (Max workspace memory, precision flags like FP16).
5.  **Build:** The compilation step.

## Step-by-Step Instructions

### Task 1: Initialize TensorRT
1.  Create `ILogger`.
2.  Create `IBuilder`.
3.  Create `INetworkDefinition` (with `kEXPLICIT_BATCH` flag).
4.  Create `IParser`.

### Task 2: Parse ONNX
1.  Load the ONNX file.
2.  `parser->parseFromFile(...)`.
3.  Check for errors.

### Task 3: Build Engine
1.  Create `IBuilderConfig`.
2.  Set memory limit (`config->setMemoryPoolLimit`).
3.  Enable FP16 (optional): `config->setFlag(BuilderFlag::kFP16)`.
4.  `builder->buildSerializedNetwork(...)`.

### Task 4: Save
1.  Write the resulting `IHostMemory` blob to a file (e.g., `model.engine`).

## Code Hints
```cpp
auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

parser->parseFromFile(onnxPath, static_cast<int>(nvinfer1::ILogger::Severity::kINFO));

auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB

auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
```

## Verification
The program should output "Engine built successfully" and create a `model.engine` file.
