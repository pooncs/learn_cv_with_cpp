cmake_minimum_required(VERSION 3.15)
project(ex09_par_algo_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(par_lib src/parallel_algos.cpp)

# Main Executable
add_executable(par_app src/main.cpp)
target_link_libraries(par_app PRIVATE par_lib)

# Tests
enable_testing()
add_executable(par_test tests/test_par.cpp)
target_link_libraries(par_test PRIVATE par_lib GTest::gtest_main)

add_test(NAME ParallelTest COMMAND par_test)
