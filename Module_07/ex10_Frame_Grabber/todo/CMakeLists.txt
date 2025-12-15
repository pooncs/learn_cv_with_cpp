cmake_minimum_required(VERSION 3.15)
project(ex10_grabber_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(grabber_lib src/frame_grabber.cpp)
target_link_libraries(grabber_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(grabber_app src/main.cpp)
target_link_libraries(grabber_app PRIVATE grabber_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(grabber_test tests/test_grabber.cpp)
target_link_libraries(grabber_test PRIVATE grabber_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME GrabberTest COMMAND grabber_test)
