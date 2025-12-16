cmake_minimum_required(VERSION 3.15)
project(ex07_preprocessing CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE OpenCV::opencv)

enable_testing()
add_executable(test_preproc tests/test_preproc.cpp)
target_link_libraries(test_preproc PRIVATE GTest::gtest_main OpenCV::opencv)
add_test(NAME test_preproc COMMAND test_preproc)
