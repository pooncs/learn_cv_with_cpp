#include <iostream>
#include <fstream>
#include <vector>

// Define this macro if TRT headers are available, else mock or fail
#if __has_include(<NvInfer.h>)
#include <NvInfer.h>
#include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only log warnings and errors
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

int main() {
    Logger logger;

    // 1. Create Builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        std::cerr << "Failed to create builder" << std::endl;
        return -1;
    }

    // 2. Create Network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        std::cerr << "Failed to create network" << std::endl;
        return -1;
    }

    // 3. Create Parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        std::cerr << "Failed to create parser" << std::endl;
        return -1;
    }

    // 4. Parse ONNX
    const char* onnx_path = "simple_model.onnx";
    // Check file existence first
    std::ifstream f(onnx_path);
    if (!f.good()) {
        std::cerr << "ONNX file not found: " << onnx_path << std::endl;
        return -1;
    }

    if (!parser->parseFromFile(onnx_path, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        return -1;
    }

    // 5. Build Config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create config" << std::endl;
        return -1;
    }
    
    // Set Memory Pool (Workspace) - e.g. 1GB
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);

    // FP16 if supported
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // 6. Build Serialized Engine
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        std::cerr << "Failed to build engine" << std::endl;
        return -1;
    }

    // 7. Save to File
    std::ofstream outfile("simple_model.engine", std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    outfile.close();

    std::cout << "Engine built and saved to simple_model.engine" << std::endl;

    return 0;
}
#else
int main() {
    std::cerr << "TensorRT headers not found. Please install TensorRT." << std::endl;
    return 0;
}
#endif
