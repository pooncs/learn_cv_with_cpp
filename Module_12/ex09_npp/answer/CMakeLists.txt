cmake_minimum_required(VERSION 3.15)
project(ex09_npp_answer LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(npp_demo src/main.cpp src/npp_resize.cpp)
target_include_directories(npp_demo PRIVATE include)

# Link against NPP libraries. In CUDAToolkit, these are CUDA::nppc, CUDA::nppi, etc.
# For resize, we usually need nppig (geometry) or just nppi (umbrella).
# CUDAToolkit usually exposes CUDA::nppi which might link all, or we specify subsets.
# Let's try CUDA::nppig CUDA::nppc CUDA::nppi
target_link_libraries(npp_demo PRIVATE CUDA::cudart CUDA::nppc CUDA::nppig CUDA::nppi)

set_target_properties(npp_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
