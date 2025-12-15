cmake_minimum_required(VERSION 3.15)
project(ex04_custom_widget_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED)

add_executable(custom_widget src/main.cpp)
target_link_libraries(custom_widget PRIVATE Qt6::Widgets)
