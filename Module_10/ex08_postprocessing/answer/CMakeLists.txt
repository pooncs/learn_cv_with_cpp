cmake_minimum_required(VERSION 3.15)
project(ex08_postprocessing_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(postproc src/main.cpp)
target_link_libraries(postproc PRIVATE OpenCV::opencv)
