cmake_minimum_required(VERSION 3.15)
project(ex09_nms_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("../../../exercise_config.cmake")

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_executable(main src/main.cpp src/nms.cpp)
target_link_libraries(main PRIVATE OpenCV::opencv)

enable_testing()
add_executable(test_nms tests/test_nms.cpp src/nms.cpp)
target_link_libraries(test_nms PRIVATE GTest::gtest_main OpenCV::opencv)
add_test(NAME test_nms COMMAND test_nms)
