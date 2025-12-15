cmake_minimum_required(VERSION 3.15)
project(ex10_3d_viewer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})

add_library(viewer src/viewer.cpp)
target_link_libraries(viewer ${OpenCV_LIBS} Eigen3::Eigen fmt::fmt)

add_executable(main src/main.cpp)
target_link_libraries(main viewer)

enable_testing()
add_executable(run_tests tests/test_viewer.cpp)
target_link_libraries(run_tests viewer GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
