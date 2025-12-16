cmake_minimum_required(VERSION 3.15)
project(ex10_3d_viewer)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(viewer_lib src/viewer.cpp)
target_link_libraries(viewer_lib PRIVATE OpenCV::OpenCV Eigen3::Eigen fmt::fmt)

add_executable(viewer src/main.cpp)
target_link_libraries(viewer viewer_lib OpenCV::OpenCV Eigen3::Eigen fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_viewer.cpp)
target_link_libraries(unit_tests_bin viewer_lib GTest::gtest_main OpenCV::OpenCV Eigen3::Eigen)
add_test(NAME UnitTests COMMAND unit_tests_bin)
