# Exercise 02: Basic Library

## Goal
Learn how to create a library (static or shared) and link it to an executable.

## Learning Objectives
1.  **Library Creation:** `add_library(lib_name STATIC source.cpp)`.
2.  **Include Directories:** `target_include_directories`.
3.  **Linking:** `target_link_libraries(app PRIVATE lib_name)`.

## Practical Motivation
Large projects are split into libraries (e.g., `core`, `utils`, `gui`) to improve organization and build times.

## Step-by-Step Instructions

### Task 1: Create a Math Library
1.  `include/my_math.hpp`: Declare `int add(int a, int b);`.
2.  `src/my_math.cpp`: Implement it.

### Task 2: Create main app
`src/main.cpp`: Include header and call `add`.

### Task 3: CMakeLists.txt
1.  `add_library(my_math src/my_math.cpp)`
2.  `target_include_directories(my_math PUBLIC include)`
3.  `add_executable(my_app src/main.cpp)`
4.  `target_link_libraries(my_app PRIVATE my_math)`

## Common Pitfalls
1.  **Visibility:** Use `PUBLIC` for include directories if consumers of the library need those headers.
2.  **Link Order:** Although Modern CMake handles transitive deps, ensure you link *to* the library *from* the executable.

## Verification
Run `my_app`. It should print the sum.
