cmake_minimum_required(VERSION 3.15)
project(ex06_inference_execution_todo CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorRT QUIET)
find_package(CUDAToolkit QUIET)

add_executable(trt_inference_todo src/main.cpp)

if (TensorRT_FOUND)
    target_link_libraries(trt_inference_todo PRIVATE TensorRT::nvinfer)
endif()

if (CUDAToolkit_FOUND)
    target_link_libraries(trt_inference_todo PRIVATE CUDA::cudart)
endif()
