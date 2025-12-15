cmake_minimum_required(VERSION 3.15)
project(ex07_ransac_todo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

# Library
add_library(ransac_lib src/robust_matcher.cpp)
target_link_libraries(ransac_lib PRIVATE ${OpenCV_LIBS})

# Main Executable
add_executable(ransac_app src/main.cpp)
target_link_libraries(ransac_app PRIVATE ransac_lib ${OpenCV_LIBS})

# Tests
enable_testing()
add_executable(ransac_test tests/test_ransac.cpp)
target_link_libraries(ransac_test PRIVATE ransac_lib GTest::gtest_main ${OpenCV_LIBS})

add_test(NAME RansacTest COMMAND ransac_test)
