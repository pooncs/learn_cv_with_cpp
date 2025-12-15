cmake_minimum_required(VERSION 3.15)
project(ex08_atomic_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(atomic_demo src/main.cpp src/atomic_counter.cpp)
