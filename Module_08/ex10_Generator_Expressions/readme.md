# Exercise 10: Generator Expressions

## Goal
Use generator expressions to customize build settings based on configuration (Debug/Release) or target properties.

## Learning Objectives
1.  **Syntax:** `$<CONDITION:VALUE>`.
2.  **Config Specific:** `$<CONFIG:Debug>` is true only when building in Debug mode.
3.  **Evaluation Time:** These are evaluated *after* CMake configure, during build file generation.

## Practical Motivation
You want to define `DEBUG_LOGGING` only for Debug builds, without using `if(CMAKE_BUILD_TYPE STREQUAL "Debug")` (which is problematic for multi-config generators like Visual Studio).

## Step-by-Step Instructions

### Task 1: main.cpp
Print "Debug Mode" if `DEBUG_BUILD` is defined, else "Release Mode".

### Task 2: CMakeLists.txt
1.  `add_executable(...)`.
2.  `target_compile_definitions(my_app PRIVATE $<$<CONFIG:Debug>:DEBUG_BUILD>)`.

### Task 3: Build
1.  `cmake ..`
2.  `cmake --build . --config Debug` -> Should print "Debug Mode".
3.  `cmake --build . --config Release` -> Should print "Release Mode".

## Common Pitfalls
1.  **Quoting:** Complex expressions might need quotes.
2.  **Support:** Not all CMake commands support generator expressions (but `target_compile_definitions/options/includes` do).

## Verification
Run both configs and verify output.
