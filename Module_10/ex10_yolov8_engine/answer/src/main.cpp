#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <algorithm>

#if __has_include(<NvInfer.h>)
#include <NvInfer.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
};

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

class YoloEngine {
public:
    YoloEngine(const std::string& engine_path) {
        // Load Engine
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.good()) throw std::runtime_error("Engine file not found");
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);

        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), size));
        context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

        // Allocate Buffers (Assuming known sizes for this demo)
        // In real app, query bindings
        input_size = 1 * 3 * 640 * 640 * sizeof(float);
        output_size = 1 * 84 * 8400 * sizeof(float); // Example YOLO output

        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_output, output_size);
        
        h_input.resize(input_size / sizeof(float));
        h_output.resize(output_size / sizeof(float));
    }

    ~YoloEngine() {
        cudaFree(d_input);
        cudaFree(d_output);
    }

    std::vector<Detection> infer(const cv::Mat& img) {
        // 1. Preprocess
        preprocess(img);

        // 2. Transfer
        cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);

        // 3. Execute
        void* bindings[] = {d_input, d_output}; // Assuming 0=in, 1=out
        context->executeV2(bindings);

        // 4. Transfer Back
        cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

        // 5. Postprocess
        return postprocess();
    }

private:
    void preprocess(const cv::Mat& img) {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(640, 640));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);
        
        int offset = 0;
        int step = 640 * 640;
        for (int c = 0; c < 3; ++c) {
            memcpy(h_input.data() + offset, channels[c].data, step * sizeof(float));
            offset += step;
        }
    }

    std::vector<Detection> postprocess() {
        // Simplified parsing for 1x84x8400
        // ... (See Ex 08)
        // Returning dummy for compilation in this stub
        return {}; 
    }

    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    void *d_input = nullptr, *d_output = nullptr;
    std::vector<float> h_input, h_output;
    size_t input_size, output_size;
};

int main() {
    try {
        YoloEngine engine("yolov8n.engine"); // Requires existing engine
        cv::Mat img = cv::imread("data/lenna.png");
        if (img.empty()) return -1;

        auto detections = engine.infer(img);
        std::cout << "Detections: " << detections.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "Note: Ensure 'yolov8n.engine' exists (generated from Ex 03)." << std::endl;
    }
    return 0;
}
#else
int main() {
    std::cout << "TensorRT not available." << std::endl;
    return 0;
}
#endif
