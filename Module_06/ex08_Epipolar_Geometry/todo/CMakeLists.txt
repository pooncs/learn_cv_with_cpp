cmake_minimum_required(VERSION 3.15)
project(ex08_epipolar_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(epipolar_lib src/epipolar.cpp)
target_link_libraries(epipolar_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(epipolar_app src/main.cpp)
target_link_libraries(epipolar_app PRIVATE epipolar_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(epipolar_test tests/test_epipolar.cpp)
target_link_libraries(epipolar_test PRIVATE epipolar_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME EpipolarTest COMMAND epipolar_test)
