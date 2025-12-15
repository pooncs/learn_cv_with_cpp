cmake_minimum_required(VERSION 3.10)
project(ex09_findmod LANGUAGES CXX)

# Tell CMake where to look for FindFoo.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Foo REQUIRED)

add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE Foo::Foo)
