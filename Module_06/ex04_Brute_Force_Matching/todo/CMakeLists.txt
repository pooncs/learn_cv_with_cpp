cmake_minimum_required(VERSION 3.15)
project(ex04_bf_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(bf_lib src/matcher.cpp)
target_link_libraries(bf_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(bf_app src/main.cpp)
target_link_libraries(bf_app PRIVATE bf_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(bf_test tests/test_matcher.cpp)
target_link_libraries(bf_test PRIVATE bf_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME BFMatcherTest COMMAND bf_test)
