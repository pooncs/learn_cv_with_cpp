cmake_minimum_required(VERSION 3.15)
project(ex06_async_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(async_lib src/async_tasks.cpp)

# Main Executable
add_executable(async_app src/main.cpp)
target_link_libraries(async_app PRIVATE async_lib)

# Tests
enable_testing()
add_executable(async_test tests/test_async.cpp)
target_link_libraries(async_test PRIVATE async_lib GTest::gtest_main)

add_test(NAME AsyncTest COMMAND async_test)
