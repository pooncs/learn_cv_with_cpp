cmake_minimum_required(VERSION 3.15)
project(ex04_normal_estimation)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(normals_lib src/normals.cpp)
target_link_libraries(normals_lib PRIVATE Eigen3::Eigen fmt::fmt)

add_executable(normal_estimation src/main.cpp)
target_link_libraries(normal_estimation normals_lib Eigen3::Eigen fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_normals.cpp)
target_link_libraries(unit_tests_bin normals_lib GTest::gtest_main Eigen3::Eigen)
add_test(NAME UnitTests COMMAND unit_tests_bin)
