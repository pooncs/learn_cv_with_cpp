cmake_minimum_required(VERSION 3.15)
project(ex09_nms_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(nms_demo src/main.cpp)
target_link_libraries(nms_demo PRIVATE OpenCV::opencv)
