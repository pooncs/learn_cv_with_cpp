cmake_minimum_required(VERSION 3.15)
project(ex03_image_display_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(image_display src/main.cpp)
target_link_libraries(image_display PRIVATE Qt6::Widgets OpenCV::opencv)
