cmake_minimum_required(VERSION 3.15)
project(ex10_calibration_tool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(calib src/Calibrator.cpp)
target_link_libraries(calib ${OpenCV_LIBS} fmt::fmt)

add_executable(main src/main.cpp)
target_link_libraries(main calib)

enable_testing()
add_executable(run_tests tests/test_calib.cpp)
target_link_libraries(run_tests calib GTest::gtest_main)
add_test(NAME UnitTests COMMAND run_tests)
