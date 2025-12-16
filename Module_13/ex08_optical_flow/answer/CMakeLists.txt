cmake_minimum_required(VERSION 3.10)
project(optical_flow_lk)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(optical_flow src/main.cpp)
target_link_libraries(optical_flow PRIVATE OpenCV::opencv)
