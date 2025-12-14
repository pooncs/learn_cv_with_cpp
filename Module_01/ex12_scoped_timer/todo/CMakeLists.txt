cmake_minimum_required(VERSION 3.15)
project(ex12_scoped_timer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

include_directories(include)

add_library(timer src/ScopedTimer.cpp)

add_executable(main src/main.cpp)
target_link_libraries(main timer)

enable_testing()
add_executable(run_tests tests/test_timer.cpp)
target_link_libraries(run_tests timer GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
