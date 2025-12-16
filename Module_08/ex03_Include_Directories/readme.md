# Module 08 - Exercise 03: Include Directories

## Goal
Understand how to manage header files and include paths properly using `target_include_directories`.

## Concept
In Modern CMake, we avoid `include_directories` (which is global) and prefer `target_include_directories` (which is scoped to a target).

### Keywords
1.  **PRIVATE**: Headers are needed to build the target, but not by consumers of the target.
2.  **INTERFACE**: Headers are needed by consumers, but not to build the target itself (header-only libs).
3.  **PUBLIC**: Headers are needed by both.

## Task
1.  Create a library `math_lib` with a header `include/math_utils.hpp` and source `src/math_utils.cpp`.
2.  The header should be in a separate `include` directory.
3.  Use `target_include_directories` to expose the `include` folder to consumers (PUBLIC).
4.  Link the library to an executable `app`.

## Instructions
1.  Navigate to `todo/`.
2.  Edit `CMakeLists.txt` to define the library and executable.
3.  Ensure `main.cpp` can include `math_utils.hpp` without specifying the full path (e.g., `#include "math_utils.hpp"`).

## Build
```bash
mkdir build
cd build
cmake ..
cmake --build .
```
