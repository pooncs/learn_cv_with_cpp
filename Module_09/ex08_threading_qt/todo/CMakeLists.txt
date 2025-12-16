cmake_minimum_required(VERSION 3.15)
project(ex08_threading_qt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets Concurrent REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE Qt6::Widgets Qt6::Concurrent)

enable_testing()
add_executable(test_threading tests/test_threading.cpp)
target_link_libraries(test_threading PRIVATE GTest::gtest_main Qt6::Widgets Qt6::Concurrent)
add_test(NAME test_threading COMMAND test_threading)
