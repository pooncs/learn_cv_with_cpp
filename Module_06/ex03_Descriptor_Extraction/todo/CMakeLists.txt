cmake_minimum_required(VERSION 3.15)
project(ex03_orb_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(orb_lib src/orb_custom.cpp)
target_link_libraries(orb_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(orb_app src/main.cpp)
target_link_libraries(orb_app PRIVATE orb_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(orb_test tests/test_orb.cpp)
target_link_libraries(orb_test PRIVATE orb_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME OrbTest COMMAND orb_test)
