cmake_minimum_required(VERSION 3.15)
project(ex02_race_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(race_lib src/counters.cpp)

# Main Executable
add_executable(race_app src/main.cpp)
target_link_libraries(race_app PRIVATE race_lib)

# Tests
enable_testing()
add_executable(race_test tests/test_race.cpp)
target_link_libraries(race_test PRIVATE race_lib GTest::gtest_main)

add_test(NAME RaceTest COMMAND race_test)
