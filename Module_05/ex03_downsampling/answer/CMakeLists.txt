cmake_minimum_required(VERSION 3.15)
project(ex03_downsampling)

set(CMAKE_CXX_STANDARD 17)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(downsample_lib src/downsample.cpp)
target_link_libraries(downsample_lib PRIVATE fmt::fmt)

add_executable(downsample src/main.cpp)
target_link_libraries(downsample downsample_lib fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_downsample.cpp)
target_link_libraries(unit_tests_bin downsample_lib GTest::gtest_main)
add_test(NAME UnitTests COMMAND unit_tests_bin)
