cmake_minimum_required(VERSION 3.15)
project(ex03_deadlock_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(deadlock_lib src/resources.cpp)

# Main Executable
add_executable(deadlock_app src/main.cpp)
target_link_libraries(deadlock_app PRIVATE deadlock_lib)

# Tests
enable_testing()
add_executable(deadlock_test tests/test_deadlock.cpp)
target_link_libraries(deadlock_test PRIVATE deadlock_lib GTest::gtest_main)

add_test(NAME DeadlockTest COMMAND deadlock_test)
