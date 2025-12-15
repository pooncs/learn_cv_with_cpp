cmake_minimum_required(VERSION 3.15)
project(ex07_pool_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(pool_demo src/main.cpp src/thread_pool.cpp)
