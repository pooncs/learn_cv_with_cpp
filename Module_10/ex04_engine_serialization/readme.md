# Exercise 04: Engine Serialization

## Goal
Load a serialized TensorRT engine (`.engine` or `.plan` file) from disk and deserialize it into an `ICudaEngine` object ready for inference.

## Learning Objectives
1.  Read a binary file into a memory buffer.
2.  Use `IRuntime` to deserialize the engine.
3.  Verify the engine dimensions/bindings.

## Practical Motivation
Building an engine (Ex 03) can take minutes. Deserializing it takes milliseconds. In production, we build once (offline) and load many times (runtime).

## Prerequisites
*   `simple_model.engine` generated from Exercise 03. (If missing, the answer code will fail gracefully).

## Theory: IRuntime
The `nvinfer1::IRuntime` interface is the entry point for deserialization. It requires a logger and the memory buffer containing the plan.

## Step-by-Step Instructions

### Task 1: Read File
Open `todo/src/main.cpp`.
1.  Open `simple_model.engine` in binary mode (`std::ios::binary`).
2.  Get file size.
3.  Read content into a `std::vector<char>` or string.

### Task 2: Deserialize
1.  Create `IRuntime`.
2.  Call `runtime->deserializeCudaEngine(data, size)`.

### Task 3: Inspect Engine
1.  Print the number of bindings (`engine->getNbBindings()`).
2.  Print the name and dimensions of input/output layers.

## Verification
Run the app. It should print "Loaded engine with X bindings".
