# Exercise 07: Subdirectories

## Goal
Structure a project hierarchically using `add_subdirectory`.

## Learning Objectives
1.  **Structure:** Moving `CMakeLists.txt` into subfolders.
2.  **Scope:** Variables set in a child directory are not visible in the parent (unless `PARENT_SCOPE` is used). Targets are visible globally.

## Practical Motivation
A clean project layout separates `src`, `tests`, `docs`, etc.

## Step-by-Step Instructions

### Task 1: Directory Layout
```
root/
  CMakeLists.txt
  src/
    CMakeLists.txt
    main.cpp
  lib/
    CMakeLists.txt
    my_lib.cpp
    my_lib.hpp
```

### Task 2: Root CMakeLists.txt
1.  `project(...)`.
2.  `add_subdirectory(lib)`.
3.  `add_subdirectory(src)`.

### Task 3: Lib CMakeLists.txt
1.  `add_library(my_lib ...)`.
2.  `target_include_directories(...)`.

### Task 4: Src CMakeLists.txt
1.  `add_executable(my_app main.cpp)`.
2.  `target_link_libraries(my_app PRIVATE my_lib)`.

## Common Pitfalls
1.  **Order:** You usually add the library subdirectory *before* the app subdirectory if the app depends on it (though CMake is often smart enough to handle order if targets are defined).
2.  **Paths:** `src/main.cpp` inside `src/CMakeLists.txt` is just referred to as `main.cpp`.

## Verification
Build from root. It should recursively build lib and app.
