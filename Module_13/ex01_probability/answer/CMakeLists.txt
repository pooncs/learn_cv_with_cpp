cmake_minimum_required(VERSION 3.15)
project(ex01_probability_answer)

set(CMAKE_CXX_STANDARD 17)

add_executable(probability_basics src/main.cpp src/probability.cpp)
target_include_directories(probability_basics PRIVATE include)
