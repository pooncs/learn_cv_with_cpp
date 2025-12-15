cmake_minimum_required(VERSION 3.15)
project(ex08_atomic_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(atomic_lib src/atomic_counter.cpp)

# Main Executable
add_executable(atomic_app src/main.cpp)
target_link_libraries(atomic_app PRIVATE atomic_lib)

# Tests
enable_testing()
add_executable(atomic_test tests/test_atomic.cpp)
target_link_libraries(atomic_test PRIVATE atomic_lib GTest::gtest_main)

add_test(NAME AtomicTest COMMAND atomic_test)
