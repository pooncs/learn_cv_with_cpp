cmake_minimum_required(VERSION 3.10)
project(ex04_linking LANGUAGES CXX)

# Lib A (Interface)
add_library(lib_a INTERFACE)
target_include_directories(lib_a INTERFACE include)

# Lib B (Static)
add_library(lib_b src/lib_b.cpp)
target_include_directories(lib_b PUBLIC include)
# Link A privately (implementation detail)
target_link_libraries(lib_b PRIVATE lib_a)

# App
add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE lib_b)
