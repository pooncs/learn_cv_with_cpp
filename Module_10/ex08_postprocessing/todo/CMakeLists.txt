cmake_minimum_required(VERSION 3.15)
project(ex08_postprocessing_todo CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(postprocess_todo src/main.cpp)
target_link_libraries(postprocess_todo PRIVATE OpenCV::opencv)
