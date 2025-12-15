# Exercise 08: Documentation Generation

## Goal
Automate the generation of API documentation from code comments using Doxygen.

## Learning Objectives
1.  **Doxygen Syntax:** Learn tags like `@brief`, `@param`, `@return`, `@see`.
2.  **Configuration:** Configure `Doxyfile` for a modern look (e.g., using `doxygen-awesome-css`).
3.  **CMake Integration:** Add a `doc` target to the build system.

## Practical Motivation
Code is read more often than it is written. Good documentation helps other developers (and future you) understand how to use your library without reading the source code implementation.

## Step-by-Step Instructions

### Task 1: Document Code
Add Doxygen comments to `include/my_lib.hpp`.
```cpp
/**
 * @brief Calculates the factorial of a number.
 * @param n The input number (must be non-negative).
 * @return The factorial of n.
 * @throws std::invalid_argument if n is negative.
 */
int factorial(int n);
```

### Task 2: Configure Doxygen
Run `doxygen -g` to generate a default `Doxyfile`.
Edit it:
- `PROJECT_NAME = "My CV Lib"`
- `INPUT = include`
- `RECURSIVE = YES`
- `GENERATE_HTML = YES`

### Task 3: CMake Target
```cmake
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()
```

### Task 4: Build Docs
Run `cmake --build build --target doc`.
Open `html/index.html`.

## Verification
Ensure the generated HTML correctly displays your classes, functions, and their descriptions.
