cmake_minimum_required(VERSION 3.15)
project(ex01_device_query_todo LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(tests)

add_executable(device_query src/main.cpp)
target_link_libraries(device_query PRIVATE CUDA::cudart)
set_target_properties(device_query PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
