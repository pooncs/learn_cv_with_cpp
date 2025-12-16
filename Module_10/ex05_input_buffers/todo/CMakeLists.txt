cmake_minimum_required(VERSION 3.15)
project(ex05_input_buffers CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE CUDA::cudart)

enable_testing()
add_executable(test_buffers tests/test_buffers.cpp)
target_link_libraries(test_buffers PRIVATE GTest::gtest_main CUDA::cudart)
add_test(NAME test_buffers COMMAND test_buffers)
