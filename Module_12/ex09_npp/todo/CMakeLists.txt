cmake_minimum_required(VERSION 3.15)
project(ex09_npp_todo LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(tests)

add_executable(npp_demo src/main.cpp src/npp_resize.cpp)
target_include_directories(npp_demo PRIVATE include)
target_link_libraries(npp_demo PRIVATE CUDA::cudart CUDA::nppc CUDA::nppig CUDA::nppi)
set_target_properties(npp_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
