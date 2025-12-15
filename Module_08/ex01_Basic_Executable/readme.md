# Exercise 01: Basic Executable

## Goal
Learn the absolute basics of a `CMakeLists.txt` file to build a simple C++ executable.

## Learning Objectives
1.  **Version Requirement:** `cmake_minimum_required(VERSION 3.10)`.
2.  **Project Definition:** `project(MyProject)`.
3.  **Target Creation:** `add_executable(my_app main.cpp)`.

## Practical Motivation
Every CMake project starts here. Without these three lines, nothing works.

## Step-by-Step Instructions

### Task 1: Create main.cpp
Write a simple "Hello, CMake!" program.

### Task 2: Create CMakeLists.txt
1.  Set minimum version (e.g., 3.10).
2.  Name your project `ex01_basic`.
3.  Add an executable target named `hello_cmake` using `main.cpp`.

### Task 3: Build
```bash
mkdir build
cd build
cmake ..
cmake --build .
./hello_cmake
```

## Common Pitfalls
1.  **Typing Error:** CMake commands are case-insensitive, but arguments (filenames) are case-sensitive.
2.  **Missing Source:** If `main.cpp` isn't found, CMake configure step fails.

## Verification
Ensure `hello_cmake` (or `hello_cmake.exe` on Windows) is created and prints the message.
