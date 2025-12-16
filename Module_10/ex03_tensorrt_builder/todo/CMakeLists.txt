cmake_minimum_required(VERSION 3.15)
project(ex03_tensorrt_builder_todo CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorRT QUIET)

add_executable(trt_builder_todo src/main.cpp)

if (TensorRT_FOUND)
    target_link_libraries(trt_builder_todo PRIVATE TensorRT::nvinfer TensorRT::nvonnxparser)
endif()
