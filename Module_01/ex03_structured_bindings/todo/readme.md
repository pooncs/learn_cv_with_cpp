# Exercise 03: Structured Bindings

## Goal
Learn to use C++17 **Structured Bindings** to unpack multiple return values from functions, iterate over maps cleaner, and decompose structures without manual field access.

## Learning Objectives
1.  Unpack `std::tuple`, `std::pair`, and `struct` into individual variables.
2.  Refactor `std::map` iterations.
3.  Understand `auto` vs `auto&` in bindings.

## Practical Motivation
In Computer Vision, functions often return multiple values:
*   `get_camera_params()` -> `(fx, fy, cx, cy)`
*   `detect_objects()` -> `(id, confidence, bounding_box)`

Before C++17, you had to use `std::tie` or access `.first` / `.second`, which is unreadable.
```cpp
// Old
auto result = get_params();
float fx = std::get<0>(result); // What is 0?
```
```cpp
// New
auto [fx, fy, cx, cy] = get_params(); // Clear!
```

## Theory
Structured binding allows you to initialize multiple entities from a single composite object.
Syntax:
```cpp
auto [var1, var2, ...] = expression;
```

### Supported Types
1.  **Arrays:** `int a[2] = {1, 2}; auto [x, y] = a;`
2.  **Tuple-like:** `std::tuple`, `std::pair`.
3.  **Structs/Classes:** All non-static data members must be public.

## Step-by-Step Instructions

### Task 1: Unpacking a Struct
Open `src/main.cpp`. You have a `Point` struct.
*   **Legacy:** `double x = p.x; double y = p.y;`
*   **Task:** Use structured binding: `auto [x, y, z] = p;`

### Task 2: Unpacking a Tuple
The function `get_config()` returns a `std::tuple<int, double, std::string>`.
*   **Task:** Call it and bind the results to `id`, `threshold`, and `filename` in one line.

### Task 3: Iterating a Map
The code iterates over `std::map<std::string, int>`.
*   **Legacy:** `it->first` (Key) and `it->second` (Value).
*   **Task:** Use `for (const auto& [name, score] : scores)`.
    *   This makes the loop body much more readable.

## Common Pitfalls
*   **Number of Elements:** You must bind *all* elements. You cannot skip variables (unlike Python's `_`).
*   **Binding by Value vs Reference:**
    *   `auto [x, y] = p;` (Copies)
    *   `auto& [x, y] = p;` (References - modifying x modifies p.x)

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
The output should print the unpacked values correctly.
