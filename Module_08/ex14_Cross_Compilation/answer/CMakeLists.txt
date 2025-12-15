cmake_minimum_required(VERSION 3.10)
project(ex14_cross LANGUAGES CXX)

add_executable(my_app src/main.cpp)

message(STATUS "Cross-compiling for: ${CMAKE_SYSTEM_NAME} on ${CMAKE_SYSTEM_PROCESSOR}")
