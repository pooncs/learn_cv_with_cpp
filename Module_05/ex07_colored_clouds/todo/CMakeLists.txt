cmake_minimum_required(VERSION 3.15)
project(ex07_colored_clouds)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_library(colorize_lib src/colorize.cpp)
target_link_libraries(colorize_lib PRIVATE OpenCV::OpenCV fmt::fmt)

add_executable(colored_clouds src/main.cpp)
target_link_libraries(colored_clouds colorize_lib OpenCV::OpenCV fmt::fmt)

enable_testing()
add_executable(unit_tests_bin tests/test_colorize.cpp)
target_link_libraries(unit_tests_bin colorize_lib GTest::gtest_main OpenCV::OpenCV)
add_test(NAME UnitTests COMMAND unit_tests_bin)
