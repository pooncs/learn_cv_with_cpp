cmake_minimum_required(VERSION 3.15)
project(ex09_charts CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets Charts REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE Qt6::Widgets Qt6::Charts)

enable_testing()
add_executable(test_charts tests/test_charts.cpp)
target_link_libraries(test_charts PRIVATE GTest::gtest_main Qt6::Widgets Qt6::Charts)
add_test(NAME test_charts COMMAND test_charts)
