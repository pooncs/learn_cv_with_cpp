cmake_minimum_required(VERSION 3.15)
project(ex06_icp_alignment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)
include_directories(${Eigen3_INCLUDE_DIRS})

add_library(icp src/icp.cpp)
target_link_libraries(icp Eigen3::Eigen fmt::fmt)

add_executable(main src/main.cpp)
target_link_libraries(main icp)

enable_testing()
add_executable(run_tests tests/test_icp.cpp)
target_link_libraries(run_tests icp GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
