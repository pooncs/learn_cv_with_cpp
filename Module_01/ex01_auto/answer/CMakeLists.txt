cmake_minimum_required(VERSION 3.15)
project(ex01_auto_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("../../../exercise_config.cmake")

find_package(GTest REQUIRED)

include_directories(include)

add_library(ex01_lib_ans src/data_utils.cpp)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ex01_lib_ans)

enable_testing()
add_executable(unit_tests test/test_auto.cpp)
target_link_libraries(unit_tests PRIVATE ex01_lib_ans GTest::gtest_main)
add_test(NAME unit_tests COMMAND unit_tests)
