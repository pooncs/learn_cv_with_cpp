cmake_minimum_required(VERSION 3.15)
project(ex02_signals_slots_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED)

add_executable(signals_slots src/main.cpp)
target_link_libraries(signals_slots PRIVATE Qt6::Widgets)
