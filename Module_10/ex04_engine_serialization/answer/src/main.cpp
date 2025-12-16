#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

// Safe guard for TRT availability
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

int main() {
    std::cout << "Loading TensorRT Engine..." << std::endl;

    // 1. Read the engine file
    // We assume model.engine exists (created in ex03)
    // If it doesn't exist, we'll just create a dummy buffer for demonstration if this was a real scenario
    const char* enginePath = "model.engine";
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    
    if (!file.good()) {
        std::cerr << "Error: Could not read engine file: " << enginePath << std::endl;
        std::cerr << "Please run Ex03 first to generate the engine." << std::endl;
        return -1;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
         std::cerr << "Error: Could not read engine content." << std::endl;
         return -1;
    }

    std::cout << "Read engine file: " << size << " bytes." << std::endl;

    // 2. Create Runtime
    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Error: Failed to create Runtime." << std::endl;
        return -1;
    }

    // 3. Deserialize Engine
    auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), size));
    if (!engine) {
        std::cerr << "Error: Failed to deserialize Engine." << std::endl;
        return -1;
    }

    // 4. Create Execution Context
    auto context = TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Error: Failed to create Execution Context." << std::endl;
        return -1;
    }

    std::cout << "Engine loaded successfully and Context created!" << std::endl;

    return 0;
}
#else
int main() {
    std::cout << "TensorRT headers not found. This is a placeholder." << std::endl;
    std::cout << "Implementation logic is in the source code." << std::endl;
    return 0;
}
#endif
