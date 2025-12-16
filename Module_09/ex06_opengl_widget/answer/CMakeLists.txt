cmake_minimum_required(VERSION 3.15)
project(ex06_opengl_widget_answer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets OpenGLWidgets REQUIRED)

add_executable(gl_widget src/main.cpp)
target_link_libraries(gl_widget PRIVATE Qt6::Widgets Qt6::OpenGLWidgets)
