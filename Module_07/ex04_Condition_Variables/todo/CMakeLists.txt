cmake_minimum_required(VERSION 3.15)
project(ex04_condvar_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(condvar_lib src/safe_queue.cpp)

# Main Executable
add_executable(condvar_app src/main.cpp)
target_link_libraries(condvar_app PRIVATE condvar_lib)

# Tests
enable_testing()
add_executable(condvar_test tests/test_condvar.cpp)
target_link_libraries(condvar_test PRIVATE condvar_lib GTest::gtest_main)

add_test(NAME SafeQueueTest COMMAND condvar_test)
