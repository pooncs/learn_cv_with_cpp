cmake_minimum_required(VERSION 3.15)
project(ex02_memory_management_todo LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(tests)

add_executable(memory_management src/main.cpp)
target_link_libraries(memory_management PRIVATE CUDA::cudart)
set_target_properties(memory_management PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
