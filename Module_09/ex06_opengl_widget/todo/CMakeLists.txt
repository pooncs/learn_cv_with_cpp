cmake_minimum_required(VERSION 3.15)
project(ex06_opengl_widget CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 COMPONENTS Widgets OpenGLWidgets REQUIRED)
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE Qt6::Widgets Qt6::OpenGLWidgets)

enable_testing()
add_executable(test_gl tests/test_gl.cpp)
target_link_libraries(test_gl PRIVATE GTest::gtest_main Qt6::Widgets Qt6::OpenGLWidgets)
add_test(NAME test_gl COMMAND test_gl)
