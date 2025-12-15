cmake_minimum_required(VERSION 3.15)
project(ex07_hungarian_todo)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(tests)

add_executable(hungarian src/main.cpp src/hungarian.cpp)
target_include_directories(hungarian PRIVATE include)
