cmake_minimum_required(VERSION 3.10)
project(pinned_memory_benchmark_todo)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(benchmark_todo src/main.cpp)
target_link_libraries(benchmark_todo PRIVATE CUDA::cudart)
