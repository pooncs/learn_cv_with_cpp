#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#if __has_include(<NvInfer.h>)
#include <NvInfer.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

int main() {
    Logger logger;
    const std::string engine_path = "simple_model.engine";

    // 1. Read File
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "Engine file not found: " << engine_path << std::endl;
        return -1;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read engine file" << std::endl;
        return -1;
    }

    // 2. Create Runtime
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Failed to create runtime" << std::endl;
        return -1;
    }

    // 3. Deserialize
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), size));
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return -1;
    }

    // 4. Inspect
    int num_bindings = engine->getNbBindings();
    std::cout << "Loaded engine with " << num_bindings << " bindings:" << std::endl;
    for (int i = 0; i < num_bindings; ++i) {
        std::cout << "  Binding " << i << ": " << engine->getBindingName(i) 
                  << " (" << (engine->bindingIsInput(i) ? "Input" : "Output") << ")" << std::endl;
        
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        std::cout << "    Dims: [";
        for (int d = 0; d < dims.nbDims; ++d) {
            std::cout << dims.d[d] << (d < dims.nbDims - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

    return 0;
}
#else
int main() {
    std::cerr << "TensorRT not installed." << std::endl;
    return 0;
}
#endif
