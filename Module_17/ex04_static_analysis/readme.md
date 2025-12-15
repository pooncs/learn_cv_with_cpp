# Exercise 04: Static Analysis with Clang-Tidy

## Goal
Integrate `clang-tidy` into the CMake build system to automatically detect code smells, bug-prone patterns, and style violations during compilation.

## Learning Objectives
1.  **Static Analysis:** Understand how static analysis tools inspect code without executing it.
2.  **CMake Integration:** Learn to enable `clang-tidy` via CMake variables.
3.  **Configuration:** Configure checks using `.clang-tidy`.
4.  **Modernization:** Use `modernize-*` checks to upgrade legacy C++ code automatically.

## Practical Motivation
Compilers catch syntax errors, but they miss logical bugs and modern best practices. Clang-tidy is like an automated code reviewer that points out:
-   "You should use `nullptr` instead of `NULL`."
-   "Pass this string by const reference to avoid a copy."
-   "This variable is initialized but never used."

## Theory: Clang-Tidy
Clang-tidy is a linter part of the LLVM project. It runs checks divided into categories like `modernize`, `performance`, `readability`, `bugprone`.

## Step-by-Step Instructions

### Task 1: Create .clang-tidy
Create a configuration file `.clang-tidy` in the root:
```yaml
Checks: 'modernize-*,performance-*,readability-*,bugprone-*'
WarningsAsErrors: '*'
```

### Task 2: CMake Integration
In `CMakeLists.txt`, set the `CMAKE_CXX_CLANG_TIDY` variable *before* defining targets.
```cmake
find_program(CLANG_TIDY "clang-tidy")
if(CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY};-checks=*")
endif()
```
Or per target:
```cmake
set_target_properties(my_target PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY}")
```

### Task 3: Fix the Code
The provided code in `todo/src/main.cpp` contains several issues (legacy casts, raw loops, by-value parameters).
Build the project. The build should fail or warn (depending on configuration) due to clang-tidy violations.
Refactor the code to satisfy the linter.

## Code Hints
-   **Finding clang-tidy:** On Windows/Visual Studio, clang-tidy might be built-in or require the LLVM toolchain.
-   **Suppressing checks:** `// NOLINT` can suppress false positives, but use sparingly.

## Verification
1.  Build -> Fails/Warns.
2.  Fix code -> Build Succeeds.
