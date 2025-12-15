cmake_minimum_required(VERSION 3.10)

# Project name and language (CXX = C++)
project(ex01_basic LANGUAGES CXX)

# Set C++ standard (good practice, though not strictly required for Hello World)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define executable target
# Syntax: add_executable(TargetName SourceFiles...)
add_executable(hello_cmake src/main.cpp)
