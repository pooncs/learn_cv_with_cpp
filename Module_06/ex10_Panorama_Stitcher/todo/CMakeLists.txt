cmake_minimum_required(VERSION 3.15)
project(ex10_stitcher_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(stitcher_lib src/stitcher.cpp)
target_link_libraries(stitcher_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(stitcher_app src/main.cpp)
target_link_libraries(stitcher_app PRIVATE stitcher_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(stitcher_test tests/test_stitcher.cpp)
target_link_libraries(stitcher_test PRIVATE stitcher_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME StitcherTest COMMAND stitcher_test)
