cmake_minimum_required(VERSION 3.15)
project(ex05_input_buffers_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(gpu_buffers src/main.cpp)
target_link_libraries(gpu_buffers PRIVATE CUDA::cudart)
