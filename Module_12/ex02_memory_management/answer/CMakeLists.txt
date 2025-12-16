cmake_minimum_required(VERSION 3.10)
project(pinned_memory_benchmark)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(benchmark src/main.cpp)
target_link_libraries(benchmark PRIVATE CUDA::cudart)
