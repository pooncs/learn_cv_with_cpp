cmake_minimum_required(VERSION 3.15)
project(ex06_inference_execution CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(TRT_INCLUDE_DIR NvInfer.h HINTS $ENV{TensorRT_ROOT} PATH_SUFFIXES include)
find_library(TRT_LIB nvinfer HINTS $ENV{TensorRT_ROOT} PATH_SUFFIXES lib lib64)

if (TRT_LIB)
    include_directories(${TRT_INCLUDE_DIR})
endif()

add_executable(main src/main.cpp)
if (TRT_LIB)
    target_link_libraries(main PRIVATE ${TRT_LIB} CUDA::cudart)
endif()

# Assume engine file exists
