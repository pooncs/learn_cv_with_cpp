cmake_minimum_required(VERSION 3.15)
project(ex07_hungarian_answer)

set(CMAKE_CXX_STANDARD 17)

add_executable(hungarian src/main.cpp src/hungarian.cpp)
target_include_directories(hungarian PRIVATE include)
