cmake_minimum_required(VERSION 3.15)
project(ex07_streams_answer LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(stream_add src/main.cpp src/stream_add.cu)
target_include_directories(stream_add PRIVATE include)
target_link_libraries(stream_add PRIVATE CUDA::cudart)
set_target_properties(stream_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
