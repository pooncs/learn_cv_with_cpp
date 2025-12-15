# Exercise 14: Cross Compilation

## Goal
Understand how to cross-compile code for a different architecture (e.g., building for ARM on x86) using a toolchain file.

## Learning Objectives
1.  **Toolchain File:** A CMake script that sets `CMAKE_SYSTEM_NAME`, `CMAKE_C_COMPILER`, etc.
2.  **Usage:** `cmake -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake ..`.
3.  **Variables:** `CMAKE_FIND_ROOT_PATH` to prevent finding host libraries.

## Practical Motivation
You are developing for a Raspberry Pi or an embedded Linux board, but you want to compile on your fast workstation.

## Step-by-Step Instructions

### Task 1: Create toolchain.cmake
Since we don't have a real cross-compiler installed, we will create a "fake" toolchain file that just sets the compiler to your system compiler but claims it's a different system (e.g., "Generic").
-   `set(CMAKE_SYSTEM_NAME Generic)`
-   `set(CMAKE_C_COMPILER gcc)` (or `cl.exe` / `clang`)

### Task 2: main.cpp
Print the architecture size (pointer size).

### Task 3: Build
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake ..
cmake --build .
```
Observe the output.

## Common Pitfalls
1.  **Finding Packages:** If you cross-compile, `find_package` might find libs on your host machine (x86) instead of the target (ARM). Use `CMAKE_FIND_ROOT_PATH`.
2.  **Running Executables:** You usually cannot run the resulting executable on the host machine (unless using QEMU).

## Verification
The CMake output should say "The CXX compiler identification is ...". The system name should be Generic.
