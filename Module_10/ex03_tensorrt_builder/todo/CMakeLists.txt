cmake_minimum_required(VERSION 3.15)
project(ex03_tensorrt_builder CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Attempt to find TensorRT
find_path(TRT_INCLUDE_DIR NvInfer.h 
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT} 
    PATH_SUFFIXES include)

find_library(TRT_LIB nvinfer 
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT} 
    PATH_SUFFIXES lib lib64)

find_library(TRT_ONNX_LIB nvonnxparser 
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT} 
    PATH_SUFFIXES lib lib64)

if (TRT_INCLUDE_DIR AND TRT_LIB AND TRT_ONNX_LIB)
    message(STATUS "Found TensorRT: ${TRT_LIB}")
    include_directories(${TRT_INCLUDE_DIR})
else()
    message(WARNING "TensorRT not found. Please set TensorRT_ROOT.")
endif()

add_executable(main src/main.cpp)
if (TRT_LIB)
    target_link_libraries(main PRIVATE ${TRT_LIB} ${TRT_ONNX_LIB} CUDA::cudart)
endif()

# Copy data
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/simple_model.onnx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
