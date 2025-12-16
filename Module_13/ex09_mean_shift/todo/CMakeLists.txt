cmake_minimum_required(VERSION 3.10)
project(mean_shift_tracker_todo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(mean_shift_todo src/main.cpp)
target_link_libraries(mean_shift_todo PRIVATE OpenCV::opencv)
