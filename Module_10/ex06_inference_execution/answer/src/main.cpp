#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <cuda_runtime.h>

#if __has_include(<NvInfer.h>)
#include <NvInfer.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};

int main() {
    Logger logger;
    // 1. Load Engine
    std::string engine_path = "simple_model.engine";
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "Engine not found." << std::endl;
        return -1;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), size));
    
    // 2. Create Context
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // 3. Allocate Buffers
    // Assume we know sizes for this example
    size_t input_bytes = 1 * 3 * 224 * 224 * sizeof(float);
    size_t output_bytes = 1 * 1000 * sizeof(float);
    
    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);

    std::vector<float> h_input(1 * 3 * 224 * 224, 1.0f);
    std::vector<float> h_output(1 * 1000);

    // 4. Bindings
    void* bindings[2];
    int input_idx = engine->getBindingIndex("input"); // Check your ONNX input name
    int output_idx = engine->getBindingIndex("output");
    
    // Fallback if names are different or using indices directly
    if (input_idx == -1) input_idx = 0; 
    if (output_idx == -1) output_idx = 1;

    bindings[input_idx] = d_input;
    bindings[output_idx] = d_output;

    // 5. Inference
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);
    
    context->executeV2(bindings);
    
    cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);

    // 6. Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    std::cout << "Inference executed successfully." << std::endl;
    return 0;
}
#else
int main() { return 0; }
#endif
