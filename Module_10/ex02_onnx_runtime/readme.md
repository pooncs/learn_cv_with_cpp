# Exercise 02: ONNX Runtime C++

## Goal
Load the ONNX model generated in Exercise 01 and run inference using the C++ ONNX Runtime API.

## Learning Objectives
1.  Initialize the ONNX Runtime Environment.
2.  Create an Inference Session.
3.  Prepare input tensors (CPU memory).
4.  Run inference and retrieve output.

## Practical Motivation
Deploying models in C++ avoids the overhead of the Python interpreter and allows integration into high-performance applications (e.g., real-time video processing).

## Prerequisites
*   You must have `simple_model.onnx` from Exercise 01. Copy it to the `data/` folder of this exercise.

## Theory: ONNX Runtime (ORT) API
*   **Env:** The global environment (threading, logging).
*   **Session:** Wrapper around the model, handles optimization and execution.
*   **MemoryInfo:** Defines where memory is allocated (CPU/GPU).
*   **Value:** Represents a tensor.

## Step-by-Step Instructions

### Task 1: Initialize ORT
Open `todo/src/main.cpp`.
1.  Include `<onnxruntime_cxx_api.h>`.
2.  Create `Ort::Env`.
3.  Create `Ort::SessionOptions`.
4.  Load the model: `Ort::Session session(env, model_path, session_options)`.

### Task 2: Prepare Input
1.  Define input shape `{1, 10}`.
2.  Create a `std::vector<float>` with 10 random values.
3.  Create an `Ort::Value` from the vector using `Ort::Value::CreateTensor`.

### Task 3: Run Inference
1.  Call `session.Run(...)`.
    *   Pass input names, input values, output names.
2.  Get the result tensor.
3.  Print the output values.

## Verification
Build and run. It should print the output tensor values (2 float values).
