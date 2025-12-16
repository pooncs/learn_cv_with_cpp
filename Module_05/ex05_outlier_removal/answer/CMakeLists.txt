cmake_minimum_required(VERSION 3.15)
project(ex05_outlier_removal)

set(CMAKE_CXX_STANDARD 17)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(sor_lib src/sor.cpp)
target_link_libraries(sor_lib PRIVATE fmt::fmt)

add_executable(outlier_removal src/main.cpp)
target_link_libraries(outlier_removal sor_lib fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_sor.cpp)
target_link_libraries(unit_tests_bin sor_lib GTest::gtest_main)
add_test(NAME UnitTests COMMAND unit_tests_bin)
