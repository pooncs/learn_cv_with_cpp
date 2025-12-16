# Exercise 03: Target Properties

## Goal
Control compilation flags and features using Modern CMake target properties, rather than global flags.

## Learning Objectives
1.  **C++ Standard:** `target_compile_features(target PUBLIC cxx_std_17)`.
2.  **Definitions:** `target_compile_definitions(target PRIVATE DEBUG_MODE)`.
3.  **Optimization:** Understanding how CMake handles `-O3` or `/O2` via `CMAKE_BUILD_TYPE`.

## Practical Motivation
You often need to enable specific C++ features (like C++17) or pass preprocessor macros (`-DDEBUG`) to specific targets without affecting others.

## Step-by-Step Instructions

### Task 1: main.cpp
Write code that uses:
1.  A C++17 feature (e.g., `std::string_view` or `if constexpr`).
2.  A preprocessor check `#ifdef MY_FEATURE`.

### Task 2: CMakeLists.txt
1.  Add executable.
2.  Use `target_compile_features` to require C++17.
3.  Use `target_compile_definitions` to define `MY_FEATURE`.

### Task 3: Build
Verify that the code compiles and the feature block is executed.

## Common Pitfalls
1.  **Global Flags:** Avoid `set(CMAKE_CXX_FLAGS "-std=c++17")`. Use target properties instead.
2.  **Public vs Private:** Use `PRIVATE` if the definition is only used inside the target's `.cpp` files. Use `PUBLIC` if it appears in the header.

## Verification
Run the executable. It should print "Feature Enabled" and use C++17 features successfully.
