cmake_minimum_required(VERSION 3.15)
project(ex01_qt_setup_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED)

add_executable(qt_hello_world src/main.cpp)
target_link_libraries(qt_hello_world PRIVATE Qt6::Widgets)
