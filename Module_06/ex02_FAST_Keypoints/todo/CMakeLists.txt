cmake_minimum_required(VERSION 3.15)
project(ex02_fast_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(fast_lib src/fast.cpp)
target_link_libraries(fast_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(fast_app src/main.cpp)
target_link_libraries(fast_app PRIVATE fast_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(fast_test tests/test_fast.cpp)
target_link_libraries(fast_test PRIVATE fast_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME FastTest COMMAND fast_test)
