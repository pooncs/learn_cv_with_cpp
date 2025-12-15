cmake_minimum_required(VERSION 3.15)
project(ex06_flann_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(flann_lib src/flann_matcher.cpp)
target_link_libraries(flann_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(flann_app src/main.cpp)
target_link_libraries(flann_app PRIVATE flann_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(flann_test tests/test_flann.cpp)
target_link_libraries(flann_test PRIVATE flann_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME FlannTest COMMAND flann_test)
