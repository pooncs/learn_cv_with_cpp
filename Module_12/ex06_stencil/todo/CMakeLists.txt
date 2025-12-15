cmake_minimum_required(VERSION 3.15)
project(ex06_stencil_todo LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(tests)

add_executable(box_filter src/main.cpp src/box_filter.cu)
target_include_directories(box_filter PRIVATE include)
target_link_libraries(box_filter PRIVATE CUDA::cudart)
set_target_properties(box_filter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
