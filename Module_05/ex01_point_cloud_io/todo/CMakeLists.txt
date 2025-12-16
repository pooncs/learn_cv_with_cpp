cmake_minimum_required(VERSION 3.15)
project(ex01_point_cloud_io)

set(CMAKE_CXX_STANDARD 17)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(point_io_lib src/PointCloudIO.cpp)
target_link_libraries(point_io_lib PRIVATE fmt::fmt)

add_executable(point_io src/main.cpp)
target_link_libraries(point_io point_io_lib fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_io.cpp)
target_link_libraries(unit_tests_bin point_io_lib GTest::gtest_main)
add_test(NAME UnitTests COMMAND unit_tests_bin)
