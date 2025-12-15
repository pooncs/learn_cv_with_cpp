# Exercise 05: Variables and Cache

## Goal
Understand the difference between normal variables (`set(VAR val)`) and cache variables (`set(VAR val CACHE ...)`, `option(...)`).

## Learning Objectives
1.  **Normal Variables:** Scoped to current directory/function.
2.  **Cache Variables:** Persist across runs, visible in `cmake-gui` or `ccmake`.
3.  **Options:** `option(ENABLE_X "Desc" ON)` is a shorthand for a boolean cache variable.

## Practical Motivation
You want to allow users to configure your build (e.g., `cmake -DENABLE_GUI=OFF ..`).

## Step-by-Step Instructions

### Task 1: main.cpp
Write code that prints "Feature Enabled" or "Feature Disabled" based on a macro.

### Task 2: CMakeLists.txt
1.  Define an option `ENABLE_MY_FEATURE` defaulting to `ON`.
2.  Print its value using `message(STATUS "Feature: ${ENABLE_MY_FEATURE}")`.
3.  If `ON`, `target_compile_definitions(app PRIVATE MY_FEATURE)`.

### Task 3: Build with Options
1.  Default: `cmake ..` -> Should be ON.
2.  Disable: `cmake -DENABLE_MY_FEATURE=OFF ..` -> Should be OFF.

## Common Pitfalls
1.  **Overwriting Cache:** `set(VAR val)` without CACHE does not update the cache, but it *hides* the cache variable in the current scope.
2.  **Force:** `set(VAR val CACHE ... FORCE)` overwrites user selection (use carefully).

## Verification
Run with and without the flag and check output.
