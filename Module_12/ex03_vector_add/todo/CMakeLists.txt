cmake_minimum_required(VERSION 3.15)
project(ex03_vector_add_todo LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(tests)

add_executable(vector_add src/main.cpp src/vector_add.cu)
target_include_directories(vector_add PRIVATE include)
target_link_libraries(vector_add PRIVATE CUDA::cudart)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
