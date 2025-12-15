cmake_minimum_required(VERSION 3.15)
project(ex01_thread_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(thread_lib src/parallel_process.cpp)
target_link_libraries(thread_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(thread_app src/main.cpp)
target_link_libraries(thread_app PRIVATE thread_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(thread_test tests/test_thread.cpp)
target_link_libraries(thread_test PRIVATE thread_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME ThreadTest COMMAND thread_test)
