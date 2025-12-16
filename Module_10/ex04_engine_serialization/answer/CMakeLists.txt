cmake_minimum_required(VERSION 3.15)
project(ex04_engine_serialization_answer CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorRT QUIET)

add_executable(trt_loader src/main.cpp)

if (TensorRT_FOUND)
    target_link_libraries(trt_loader PRIVATE TensorRT::nvinfer)
else()
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        target_link_libraries(trt_loader PRIVATE CUDA::cudart)
    endif()
endif()
