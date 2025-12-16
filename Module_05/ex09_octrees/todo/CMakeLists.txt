cmake_minimum_required(VERSION 3.15)
project(ex09_octrees)

set(CMAKE_CXX_STANDARD 17)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(octree_lib src/octree.cpp)
target_link_libraries(octree_lib PRIVATE fmt::fmt)

add_executable(octrees src/main.cpp)
target_link_libraries(octrees octree_lib fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_octree.cpp)
target_link_libraries(unit_tests_bin octree_lib GTest::gtest_main)
add_test(NAME UnitTests COMMAND unit_tests_bin)
