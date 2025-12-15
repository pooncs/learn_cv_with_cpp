cmake_minimum_required(VERSION 3.15)
project(ex03_deadlock_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(deadlock_demo src/main.cpp src/resources.cpp)
