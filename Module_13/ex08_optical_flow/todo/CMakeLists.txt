cmake_minimum_required(VERSION 3.10)
project(optical_flow_lk_todo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(optical_flow_todo src/main.cpp)
target_link_libraries(optical_flow_todo PRIVATE OpenCV::opencv)
