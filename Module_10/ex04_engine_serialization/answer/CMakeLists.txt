cmake_minimum_required(VERSION 3.15)
project(ex04_engine_serialization_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(TRT_INCLUDE_DIR NvInfer.h HINTS $ENV{TensorRT_ROOT} PATH_SUFFIXES include)
find_library(TRT_LIB nvinfer HINTS $ENV{TensorRT_ROOT} PATH_SUFFIXES lib lib64)

add_executable(trt_loader src/main.cpp)
if (TRT_LIB)
    target_include_directories(trt_loader PRIVATE ${TRT_INCLUDE_DIR})
    target_link_libraries(trt_loader PRIVATE ${TRT_LIB} CUDA::cudart)
endif()
