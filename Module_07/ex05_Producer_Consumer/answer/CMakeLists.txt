cmake_minimum_required(VERSION 3.15)
project(ex05_pipeline_answer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(pipeline_demo src/main.cpp src/pipeline.cpp)
