cmake_minimum_required(VERSION 3.15)
project(ex02_ci CXX)

find_package(GTest REQUIRED)

add_library(calc INTERFACE)
target_include_directories(calc INTERFACE src)

enable_testing()
add_executable(test_calc test/test_calc.cpp)
target_link_libraries(test_calc PRIVATE calc GTest::gtest GTest::gtest_main)
add_test(NAME CalcTest COMMAND test_calc)
