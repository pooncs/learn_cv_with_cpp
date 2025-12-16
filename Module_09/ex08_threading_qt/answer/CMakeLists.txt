cmake_minimum_required(VERSION 3.15)
project(ex08_threading_qt_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets Concurrent REQUIRED)

add_executable(threading_qt src/main.cpp)
target_link_libraries(threading_qt PRIVATE Qt6::Widgets Qt6::Concurrent)
