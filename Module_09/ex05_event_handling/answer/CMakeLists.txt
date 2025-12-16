cmake_minimum_required(VERSION 3.15)
project(ex05_event_handling_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED)

add_executable(event_handling src/main.cpp)
target_link_libraries(event_handling PRIVATE Qt6::Widgets)
