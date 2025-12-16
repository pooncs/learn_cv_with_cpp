cmake_minimum_required(VERSION 3.15)
project(ex09_nms_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

# Main Application
add_executable(nms_demo src/main.cpp src/nms.cpp)
target_link_libraries(nms_demo PRIVATE OpenCV::opencv)

# Testing
enable_testing()
add_executable(nms_test tests/test_nms.cpp src/nms.cpp)
target_link_libraries(nms_test PRIVATE OpenCV::opencv)

add_test(NAME IoUTest COMMAND nms_test iou)
add_test(NAME NMSTest COMMAND nms_test nms)
