cmake_minimum_required(VERSION 3.15)
project(ex10_bilateral_answer LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(bilateral src/main.cpp src/bilateral.cu)
target_include_directories(bilateral PRIVATE include)
target_link_libraries(bilateral PRIVATE CUDA::cudart)
set_target_properties(bilateral PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
