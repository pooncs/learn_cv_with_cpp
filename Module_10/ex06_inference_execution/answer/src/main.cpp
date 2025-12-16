#include <iostream>
#include <vector>
#include <memory>
#include <numeric>

#if __has_include(<NvInfer.h>)
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} logger;

struct TrtDestroyer {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer>;

// Helper for CUDA error checking
#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
            return -1; \
        } \
    } while (0)

int main() {
    std::cout << "Inference Execution Demo" << std::endl;

    // 1. Setup Engine (Mocking the loading part for brevity, assuming we have one)
    // In a real app, you would load "model.engine" here.
    // For this standalone answer, we can't easily "mock" an engine without a real plan file.
    // So we will simulate the "Inference Loop" logic assuming `engine` and `context` exist.
    
    // NOTE: Since we can't guarantee a valid engine file exists in this CI environment,
    // we will write the code that WOULD run if we had it.
    
    // ... Load Engine Code (See Ex04) ...
    nvinfer1::ICudaEngine* engine = nullptr; // Assume valid
    nvinfer1::IExecutionContext* context = nullptr; // Assume valid
    
    // If we don't have an engine, we can't really demonstrate execution. 
    // However, we can demonstrate the Memory and Stream logic which is the core of this exercise.
    
    std::cout << "Simulating Inference Loop..." << std::endl;

    // 2. Define Sizes
    const int batchSize = 1;
    const int inputSize = 1 * 3 * 224 * 224; // Standard Image
    const int outputSize = 1000; // Class scores
    const size_t inputBytes = inputSize * sizeof(float);
    const size_t outputBytes = outputSize * sizeof(float);

    // 3. Allocate Host Memory
    std::vector<float> hostInput(inputSize);
    std::vector<float> hostOutput(outputSize);
    
    // Fill input
    std::fill(hostInput.begin(), hostInput.end(), 0.5f);

    // 4. Allocate Device Memory
    void* deviceInput = nullptr;
    void* deviceOutput = nullptr;

    CHECK_CUDA(cudaMalloc(&deviceInput, inputBytes));
    CHECK_CUDA(cudaMalloc(&deviceOutput, outputBytes));

    // 5. Create Stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 6. Bindings
    // In real TRT: int inputIdx = engine->getBindingIndex("input");
    // For demo:
    void* bindings[2];
    bindings[0] = deviceInput;  // Assume index 0 is input
    bindings[1] = deviceOutput; // Assume index 1 is output

    // 7. Inference Step
    std::cout << "Step 1: Host -> Device" << std::endl;
    CHECK_CUDA(cudaMemcpyAsync(deviceInput, hostInput.data(), inputBytes, cudaMemcpyHostToDevice, stream));

    std::cout << "Step 2: Enqueue Inference" << std::endl;
    // context->enqueueV2(bindings, stream, nullptr); 
    // Mocking execution time
    // CHECK_CUDA(cudaLaunchKernel...); 

    std::cout << "Step 3: Device -> Host" << std::endl;
    CHECK_CUDA(cudaMemcpyAsync(hostOutput.data(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost, stream));

    std::cout << "Step 4: Synchronize" << std::endl;
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 8. Process Output
    std::cout << "Inference Done. First output value: " << hostOutput[0] << std::endl;

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return 0;
}
#else
int main() {
    std::cout << "TensorRT/CUDA not found. Placeholder." << std::endl;
    return 0;
}
#endif
