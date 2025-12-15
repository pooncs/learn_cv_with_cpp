# Exercise 06: Globbing vs Explicit Lists

## Goal
Understand how to collect source files automatically using `file(GLOB ...)` and why explicit listing is generally preferred.

## Learning Objectives
1.  **Globbing:** `file(GLOB SOURCES "src/*.cpp")`.
2.  **Recursive Glob:** `file(GLOB_RECURSE ...)` searches subdirectories.
3.  **The "Evil" of Globbing:** CMake cannot detect when you add a new file unless you re-run `cmake`.
4.  **CONFIGURE_DEPENDS:** (CMake 3.12+) Hints CMake to re-run if the directory content changes.

## Practical Motivation
Listing 100 files manually is tedious. Globbing is fast but can lead to build errors if you forget to re-run CMake after adding a file.

## Step-by-Step Instructions

### Task 1: Create Multiple Sources
1.  `src/main.cpp`
2.  `src/part1.cpp`
3.  `src/part2.cpp`

### Task 2: CMakeLists.txt with Glob
1.  Use `file(GLOB_RECURSE SRC_FILES "src/*.cpp")`.
2.  `add_executable(my_app ${SRC_FILES})`.

## Common Pitfalls
1.  **Adding a file:** If you create `src/part3.cpp`, `make` (or `ninja`) won't know about it until you run `cmake ..` again.
2.  **Best Practice:** For long-term projects, list files explicitly or use `CONFIGURE_DEPENDS` if available.

## Verification
1.  Build the project.
2.  Add a new file (that contains a function used by main).
3.  Run build. It might fail linking (undefined symbol) because the new file wasn't compiled.
4.  Touch `CMakeLists.txt` or run `cmake ..`.
5.  Build again. It works.
