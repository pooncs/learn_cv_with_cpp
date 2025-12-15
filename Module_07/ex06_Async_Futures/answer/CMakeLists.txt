cmake_minimum_required(VERSION 3.15)
project(ex06_async_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(async_demo src/main.cpp src/async_tasks.cpp)
