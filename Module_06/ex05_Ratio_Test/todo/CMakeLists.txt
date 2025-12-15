cmake_minimum_required(VERSION 3.15)
project(ex05_ratio_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(ratio_lib src/ratio_test.cpp)
target_link_libraries(ratio_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(ratio_app src/main.cpp)
target_link_libraries(ratio_app PRIVATE ratio_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(ratio_test tests/test_ratio.cpp)
target_link_libraries(ratio_test PRIVATE ratio_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME RatioTest COMMAND ratio_test)
