cmake_minimum_required(VERSION 3.15)
project(ex09_octrees)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(octree src/octree.cpp)
target_link_libraries(octree fmt::fmt)

add_executable(main src/main.cpp)
target_link_libraries(main octree)

enable_testing()
add_executable(run_tests tests/test_octree.cpp)
target_link_libraries(run_tests octree GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
