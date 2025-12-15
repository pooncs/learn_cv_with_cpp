cmake_minimum_required(VERSION 3.15)
project(ex02_kalman_1d_answer)

set(CMAKE_CXX_STANDARD 17)

add_executable(kalman_1d src/main.cpp src/kalman_1d.cpp)
target_include_directories(kalman_1d PRIVATE include)
