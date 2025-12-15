cmake_minimum_required(VERSION 3.10)
project(ex10_genex LANGUAGES CXX)

add_executable(my_app src/main.cpp)

# Define DEBUG_BUILD only if configuration is Debug
target_compile_definitions(my_app PRIVATE $<$<CONFIG:Debug>:DEBUG_BUILD>)
