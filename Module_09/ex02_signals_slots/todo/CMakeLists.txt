cmake_minimum_required(VERSION 3.15)
project(ex02_signals_slots CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE Qt6::Widgets)

enable_testing()
add_executable(test_signals tests/test_signals.cpp)
target_link_libraries(test_signals PRIVATE GTest::gtest_main Qt6::Widgets)
add_test(NAME test_signals COMMAND test_signals)
