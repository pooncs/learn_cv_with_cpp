cmake_minimum_required(VERSION 3.15)
project(ex05_pipeline_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(pipeline_lib src/pipeline.cpp)

# Main Executable
add_executable(pipeline_app src/main.cpp)
target_link_libraries(pipeline_app PRIVATE pipeline_lib)

# Tests
enable_testing()
add_executable(pipeline_test tests/test_pipeline.cpp)
target_link_libraries(pipeline_test PRIVATE pipeline_lib GTest::gtest_main)

add_test(NAME PipelineTest COMMAND pipeline_test)
