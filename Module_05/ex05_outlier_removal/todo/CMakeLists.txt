cmake_minimum_required(VERSION 3.15)
project(ex05_outlier_removal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(sor src/sor.cpp)
target_link_libraries(sor fmt::fmt)

add_executable(main src/main.cpp)
target_link_libraries(main sor)

enable_testing()
add_executable(run_tests tests/test_sor.cpp)
target_link_libraries(run_tests sor GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
