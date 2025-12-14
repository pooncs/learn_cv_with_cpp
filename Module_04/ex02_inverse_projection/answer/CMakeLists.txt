cmake_minimum_required(VERSION 3.15)
project(ex02_inverse_projection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(inverse_proj src/inverse_projection.cpp)
target_link_libraries(inverse_proj ${OpenCV_LIBS} fmt::fmt)

add_executable(main src/main.cpp)
target_link_libraries(main inverse_proj)

enable_testing()
add_executable(run_tests tests/test_inverse.cpp)
target_link_libraries(run_tests inverse_proj GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
