cmake_minimum_required(VERSION 3.15)
project(ex01_harris_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(harris_lib src/harris.cpp)
target_link_libraries(harris_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(harris_app src/main.cpp)
target_link_libraries(harris_app PRIVATE harris_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(harris_test tests/test_harris.cpp)
target_link_libraries(harris_test PRIVATE harris_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME HarrisTest COMMAND harris_test)
