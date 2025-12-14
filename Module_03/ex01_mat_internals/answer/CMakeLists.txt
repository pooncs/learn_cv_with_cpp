cmake_minimum_required(VERSION 3.15)
project(ex01_mat_internals)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(mat_info src/mat_info.cpp)
target_link_libraries(mat_info ${OpenCV_LIBS})

add_executable(main src/main.cpp)
target_link_libraries(main mat_info ${OpenCV_LIBS})
