cmake_minimum_required(VERSION 3.15)
project(ex13_minimal_logger)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

add_library(logger src/Logger.cpp)

add_executable(main src/main.cpp)
target_link_libraries(main logger)

enable_testing()
add_executable(run_tests tests/test_logger.cpp)
target_link_libraries(run_tests logger GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
