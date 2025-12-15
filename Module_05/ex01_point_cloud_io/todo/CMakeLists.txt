cmake_minimum_required(VERSION 3.15)
project(ex01_point_cloud_io)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(pc_io src/PointCloudIO.cpp)
target_link_libraries(pc_io fmt::fmt)

add_executable(main src/main.cpp)
target_link_libraries(main pc_io)

enable_testing()
add_executable(run_tests tests/test_io.cpp)
target_link_libraries(run_tests pc_io GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
