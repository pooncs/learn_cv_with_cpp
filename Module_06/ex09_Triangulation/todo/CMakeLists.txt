cmake_minimum_required(VERSION 3.15)
project(ex09_triangulation_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(triangulation_lib src/triangulation.cpp)
target_link_libraries(triangulation_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(triangulation_app src/main.cpp)
target_link_libraries(triangulation_app PRIVATE triangulation_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(triangulation_test tests/test_triangulation.cpp)
target_link_libraries(triangulation_test PRIVATE triangulation_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME TriangulationTest COMMAND triangulation_test)
