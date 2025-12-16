cmake_minimum_required(VERSION 3.10)
project(mean_shift_tracker)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(mean_shift src/main.cpp)
target_link_libraries(mean_shift PRIVATE OpenCV::opencv)
