#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <vector>

TEST(ONNXRuntime, LoadModel) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestONNX");
    Ort::SessionOptions session_options;
    
#ifdef _WIN32
    const wchar_t* model_path = L"simple_model.onnx";
#else
    const char* model_path = "simple_model.onnx";
#endif

    // We expect the model file to exist (copied by CMake)
    try {
        Ort::Session session(env, model_path, session_options);
        EXPECT_NE(session, nullptr);
    } catch (const Ort::Exception& e) {
        FAIL() << "Failed to load model: " << e.what();
    }
}

TEST(ONNXRuntime, RunInference) {
     // Similar to main, just verifying it runs without throwing
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestONNX");
    Ort::SessionOptions session_options;
#ifdef _WIN32
    const wchar_t* model_path = L"simple_model.onnx";
#else
    const char* model_path = "simple_model.onnx";
#endif

    try {
        Ort::Session session(env, model_path, session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        std::vector<int64_t> input_shape = {1, 10};
        std::vector<float> input_values(10, 1.0f);
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size()
        );

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        EXPECT_EQ(output_tensors.size(), 1);
        
        auto info = output_tensors[0].GetTensorTypeAndShapeInfo();
        EXPECT_EQ(info.GetShape()[0], 1);
        EXPECT_EQ(info.GetShape()[1], 2); // Output size 2
        
    } catch (...) {
        FAIL() << "Inference failed";
    }
}
