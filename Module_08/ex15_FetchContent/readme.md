# Exercise 15: FetchContent

## Goal
Use `FetchContent` to download and link a library (e.g., `fmt` or `googletest`) at configure time, without needing it installed on the system.

## Learning Objectives
1.  **FetchContent:** `FetchContent_Declare` and `FetchContent_MakeAvailable`.
2.  **Dependencies:** Automatically handling 3rd party code.
3.  **Modern CMake:** The preferred way over `ExternalProject_Add` for many cases.

## Practical Motivation
You want to use `fmt` library, but you don't want to force the user to install it manually.

## Step-by-Step Instructions

### Task 1: main.cpp
Use `fmt::print("Hello, {}!", "World");`.

### Task 2: CMakeLists.txt
1.  `include(FetchContent)`.
2.  `FetchContent_Declare(fmt ...)` (Git repo: `https://github.com/fmtlib/fmt.git`, Tag: `10.1.1`).
3.  `FetchContent_MakeAvailable(fmt)`.
4.  `add_executable(...)`.
5.  `target_link_libraries(my_app PRIVATE fmt::fmt)`.

### Task 3: Build
CMake will download `fmt` during configuration.

## Common Pitfalls
1.  **Download Time:** Configuration takes longer the first time.
2.  **Name Conflicts:** Ensure the target name (e.g., `fmt::fmt`) matches what the library provides.

## Verification
Run the app. It should print "Hello, World!".
