# Exercise 16: CMake Presets

## Goal
Use `CMakePresets.json` to define standard configure and build presets.

## Learning Objectives
1.  **Standardization:** Presets allow sharing build configurations (Debug, Release, Sanitizers) across the team.
2.  **Usage:** `cmake --preset debug` and `cmake --build --preset debug`.
3.  **JSON Structure:** Understanding `configurePresets` and `buildPresets`.

## Practical Motivation
Instead of remembering `cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON ..`, you just type `cmake --preset debug`.

## Step-by-Step Instructions

### Task 1: main.cpp
Simple hello world.

### Task 2: CMakeLists.txt
Basic project.

### Task 3: CMakePresets.json
Create a file with:
-   `configurePresets`: "default" (Release), "debug" (Debug).
-   `buildPresets`: "default", "debug".

### Task 4: Run
```bash
cmake --preset debug
cmake --build --preset debug
```

## Common Pitfalls
1.  **Version:** Presets require CMake 3.19+.
2.  **Inheritance:** You can inherit presets to avoid duplication.

## Verification
Run the presets and verify the build type (e.g., check if debug symbols are present or simply check the output directory structure).
