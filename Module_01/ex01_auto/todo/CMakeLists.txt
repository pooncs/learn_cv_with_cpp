cmake_minimum_required(VERSION 3.15)
project(ex01_auto_todo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include common configuration for 3rdParty libs
include("../../../exercise_config.cmake")

# Find GTest (provided by Conan)
find_package(GTest REQUIRED)

include_directories(include)

# Library for logic
add_library(ex01_lib src/data_utils.cpp)

# Main Executable
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ex01_lib)

# Tests
enable_testing()
add_executable(unit_tests test/test_auto.cpp)
target_link_libraries(unit_tests PRIVATE ex01_lib GTest::gtest_main)
add_test(NAME unit_tests COMMAND unit_tests)
