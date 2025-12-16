cmake_minimum_required(VERSION 3.15)
project(ex03_tensorrt_builder_answer CXX)

set(CMAKE_CXX_STANDARD 17)

# Find TensorRT - This is often environment specific
# We can try to find it or let the user provide paths
find_package(TensorRT QUIET)

if (NOT TensorRT_FOUND)
    message(STATUS "TensorRT not found via find_package. Attempting to look in standard locations or using placeholders.")
    # In a real project, you would add include directories and link libraries manually if find_package fails
    # include_directories(/usr/local/cuda/include /usr/include/x86_64-linux-gnu)
    # link_directories(/usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
endif()

add_executable(trt_builder src/main.cpp)

if (TensorRT_FOUND)
    target_link_libraries(trt_builder PRIVATE TensorRT::nvinfer TensorRT::nvonnxparser)
else()
    # Link CUDA if available, but otherwise just build the placeholder
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        target_link_libraries(trt_builder PRIVATE CUDA::cudart)
    endif()
endif()
