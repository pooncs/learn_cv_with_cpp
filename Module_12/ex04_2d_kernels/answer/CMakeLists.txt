cmake_minimum_required(VERSION 3.15)
project(ex04_2d_kernels_answer LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
# Removed OpenCV dependency for simplicity in this specific test
# find_package(OpenCV REQUIRED)

add_executable(invert_image src/main.cpp src/image_invert.cu)
target_include_directories(invert_image PRIVATE include)
target_link_libraries(invert_image PRIVATE CUDA::cudart)
set_target_properties(invert_image PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
