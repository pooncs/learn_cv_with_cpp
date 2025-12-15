cmake_minimum_required(VERSION 3.15)
project(ex05_shared_memory_answer LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(matrix_mul src/main.cpp src/matrix_mul.cu)
target_include_directories(matrix_mul PRIVATE include)
target_link_libraries(matrix_mul PRIVATE CUDA::cudart)
set_target_properties(matrix_mul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
