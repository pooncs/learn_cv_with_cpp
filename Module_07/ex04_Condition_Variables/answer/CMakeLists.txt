cmake_minimum_required(VERSION 3.15)
project(ex04_condvar_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(condvar_demo src/main.cpp src/safe_queue.cpp)
