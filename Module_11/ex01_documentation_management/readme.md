# Exercise 01: Documentation Management

## Goal
Generate comprehensive API documentation for your C++ project using Doxygen.

## Learning Objectives
1.  **Doxygen Syntax:** Learn tags like `@brief`, `@param`, `@return`.
2.  **Configuration:** Configure `Doxyfile`.
3.  **CMake Integration:** Add a `doc` target.

## Practical Motivation
Undocumented code is legacy code the moment it is written.

## Step-by-Step Instructions
1.  Add Doxygen comments to your headers.
2.  Run `doxygen -g` to create a Doxyfile.
3.  Modify Doxyfile (INPUT, RECURSIVE, OUTPUT_DIRECTORY).
4.  Add `add_custom_target(doc ...)` in CMake.

## Verification
Open `html/index.html` and verify your classes appear.
