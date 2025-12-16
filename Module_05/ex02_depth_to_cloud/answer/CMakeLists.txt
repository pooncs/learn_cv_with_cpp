cmake_minimum_required(VERSION 3.15)
project(ex02_depth_to_cloud)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(depth_util_lib src/depth_util.cpp)
target_link_libraries(depth_util_lib PRIVATE OpenCV::OpenCV fmt::fmt)

add_executable(depth_to_cloud src/main.cpp)
target_link_libraries(depth_to_cloud depth_util_lib OpenCV::OpenCV fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_depth.cpp)
target_link_libraries(unit_tests_bin depth_util_lib GTest::gtest_main OpenCV::OpenCV)
add_test(NAME UnitTests COMMAND unit_tests_bin)
