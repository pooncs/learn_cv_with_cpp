cmake_minimum_required(VERSION 3.10)
project(sort_tracker)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(sort_tracker src/main.cpp)
target_link_libraries(sort_tracker PRIVATE OpenCV::opencv)
