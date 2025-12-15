#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <numeric>

int main() {
    try {
        // 1. Setup Environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestONNX");

        // 2. Setup Session Options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        // 3. Load Model
        // Assuming simple_model.onnx is in the current directory
#ifdef _WIN32
        const wchar_t* model_path = L"simple_model.onnx";
#else
        const char* model_path = "simple_model.onnx";
#endif
        Ort::Session session(env, model_path, session_options);

        // 4. Prepare Inputs
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Define Input/Output Names
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        // Input Data
        std::vector<int64_t> input_shape = {1, 10};
        std::vector<float> input_values(10);
        std::iota(input_values.begin(), input_values.end(), 0.0f); // 0, 1, ..., 9

        // Create Tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            input_values.data(), 
            input_values.size(), 
            input_shape.data(), 
            input_shape.size()
        );

        // 5. Run Inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            &input_tensor, 
            1, 
            output_names, 
            1
        );

        // 6. Process Output
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::cout << "Output: ";
        for (size_t i = 0; i < count; i++) {
            std::cout << floatarr[i] << " ";
        }
        std::cout << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
