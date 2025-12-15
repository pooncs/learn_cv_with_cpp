cmake_minimum_required(VERSION 3.15)
project(ex08_thrust_answer LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(thrust_demo src/main.cpp src/thrust_demo.cu)
target_include_directories(thrust_demo PRIVATE include)
target_link_libraries(thrust_demo PRIVATE CUDA::cudart)
set_target_properties(thrust_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
