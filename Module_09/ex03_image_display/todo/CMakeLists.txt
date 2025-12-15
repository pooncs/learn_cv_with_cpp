cmake_minimum_required(VERSION 3.15)
project(ex03_image_display CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE Qt6::Widgets OpenCV::opencv)

enable_testing()
add_executable(test_image tests/test_image.cpp)
target_link_libraries(test_image PRIVATE GTest::gtest_main Qt6::Widgets OpenCV::opencv)
add_test(NAME test_image COMMAND test_image)
