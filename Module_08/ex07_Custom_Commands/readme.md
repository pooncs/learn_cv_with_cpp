# Module 08 - Exercise 07: Custom Commands

## Goal
Learn how to run arbitrary commands during the build process using `add_custom_command`.

## Concept
Sometimes you need to generate source code (e.g., from protobufs), copy assets, or embed build metadata (like git hash or version) before compilation.

`add_custom_command` can be used to:
1.  **Generate a file**: If a target depends on a file that doesn't exist, CMake looks for a custom command that produces it.
2.  **Run events**: `PRE_BUILD`, `PRE_LINK`, `POST_BUILD` hooks on targets.

## Task
1.  We want `main.cpp` to include a file `version.h` that defines `#define PROJECT_VERSION "1.0.0"`.
2.  This file `version.h` should be **generated** by CMake at build time, not checked into git.
3.  Use `add_custom_command` to create `version.h` before compiling `main.cpp`.

## Instructions
1.  Navigate to `todo/`.
2.  Edit `CMakeLists.txt`.
3.  Add a custom command that writes to `${CMAKE_BINARY_DIR}/generated/version.h`.
4.  Make sure the executable depends on this file (or simply add the generated file to the `add_executable` list).
5.  Don't forget to include the binary directory in search paths!

## Build
```bash
mkdir build
cd build
cmake ..
cmake --build .
```
