cmake_minimum_required(VERSION 3.15)
project(ex06_inference_execution_answer CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorRT QUIET)
find_package(CUDAToolkit QUIET)

add_executable(trt_inference src/main.cpp)

if (TensorRT_FOUND)
    target_link_libraries(trt_inference PRIVATE TensorRT::nvinfer)
endif()

if (CUDAToolkit_FOUND)
    target_link_libraries(trt_inference PRIVATE CUDA::cudart)
endif()

# Testing
enable_testing()
add_executable(test_inference tests/test_inference.cpp)
add_test(NAME InferenceLogicTest COMMAND test_inference)
