# Exercise 04: Engine Serialization & Deserialization

## Goal
Load (deserialize) a saved TensorRT engine from disk and prepare it for inference.

## Learning Objectives
1.  **Reading Binary Files:** Load the `.plan` or `.engine` file into memory.
2.  **Runtime Initialization:** Create `nvinfer1::IRuntime`.
3.  **Deserialization:** Use `runtime->deserializeCudaEngine(...)`.
4.  **Execution Context:** Create `IExecutionContext` from the engine.

## Practical Motivation
You don't want to compile the model every time your application starts (compilation can take minutes). You build it once (Ex03), save it, and then load it quickly (Ex04) at runtime.

**Analogy:**
*   **Building (Ex03):** Compiling source code into an executable `.exe`.
*   **Deserializing (Ex04):** Double-clicking the `.exe` to run it. You don't recompile `Call of Duty` every time you want to play; you just launch the installed game.

## Theory: The Runtime
*   **IRuntime:** The interface for deserializing engines. It doesn't need the network definition or parser anymore.
*   **ICudaEngine:** The optimized engine object. It holds the weights and the compiled graph.
*   **IExecutionContext:** The actual worker that runs inference. You can have multiple contexts for one engine (e.g., for multi-threaded inference).

## Step-by-Step Instructions

### Task 1: Read File
1.  Open the file in binary mode (`std::ios::binary`).
2.  Get the size.
3.  Read the content into a `std::vector<char>` or string.

### Task 2: Deserialize
1.  Create `nvinfer1::ILogger`.
2.  Create `nvinfer1::IRuntime`.
3.  Call `runtime->deserializeCudaEngine(data, size)`.

### Task 3: Create Context
1.  Call `engine->createExecutionContext()`.
2.  Verify it's not null.

## Code Hints
```cpp
// Read file
std::ifstream file("model.engine", std::ios::binary | std::ios::ate);
std::streamsize size = file.tellg();
file.seekg(0, std::ios::beg);
std::vector<char> buffer(size);
file.read(buffer.data(), size);

// Deserialize
auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), size));
auto context = TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
```

## Verification
Output "Engine loaded successfully" if the engine and context are created without errors.
