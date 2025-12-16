cmake_minimum_required(VERSION 3.15)
project(ex06_icp_alignment)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(icp_lib src/icp.cpp)
target_link_libraries(icp_lib PRIVATE Eigen3::Eigen fmt::fmt)

add_executable(icp_alignment src/main.cpp)
target_link_libraries(icp_alignment icp_lib Eigen3::Eigen fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_icp.cpp)
target_link_libraries(unit_tests_bin icp_lib GTest::gtest_main Eigen3::Eigen)
add_test(NAME UnitTests COMMAND unit_tests_bin)
