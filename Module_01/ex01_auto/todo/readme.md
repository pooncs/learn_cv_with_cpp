# Exercise 01: Auto & Type Inference

## Goal
Master the usage of `auto` and `decltype` to write cleaner, more maintainable code without sacrificing type safety. You will refactor legacy C++ code (pre-C++11 style) into modern C++17.

## Learning Objectives
1.  Understand how `auto` deduces types from initializers.
2.  Learn the difference between `auto`, `auto&`, and `const auto&`.
3.  Use `decltype` to inspect the type of an expression at compile time.
4.  Recognize when *not* to use `auto` (readability vs. brevity).

## Practical Motivation
In Computer Vision, we often deal with complex types like `std::vector<std::vector<cv::Point2f>>` (a list of contours) or nested map iterators. Writing these types manually is:
1.  **Error-prone:** Typos are common.
2.  **Hard to refactor:** Changing the container type requires updating every iterator declaration.
3.  **Verbose:** It clutters the logic.

Modern C++ allows the compiler to deduce these types for you, making the code robust to changes.

## Theory: How `auto` Works
`auto` uses the same rules as template argument deduction.

### 1. Basic Value Deduction
```cpp
int x = 5;
auto y = x; // y is int (copy of x)
```

### 2. References and Pointers
```cpp
std::vector<int> big_data;
auto  a = big_data; // COPIES the vector (expensive!)
auto& b = big_data; // Reference to original
const auto& c = big_data; // Read-only reference (Best for loops)
```

### 3. Iterators
**Old Way:**
```cpp
std::map<std::string, float>::iterator it = my_map.begin();
```
**New Way:**
```cpp
auto it = my_map.begin();
```

## Step-by-Step Instructions

### Task 1: Refactor Variable Declarations
Open `src/main.cpp`. You will see a function `get_data()` returning a complex map.
*   **Legacy:** `std::map<std::string, std::vector<int>> data = get_data();`
*   **Task:** Replace the type with `auto`.

### Task 2: Modernize Loops
The current code uses a raw `for` loop or a verbose iterator loop.
*   **Task:** Convert it to a range-based for loop.
*   **Tip:** Use `const auto&` to avoid copying the map elements (pairs).
    ```cpp
    for (const auto& pair : data) {
        // pair.first is key, pair.second is value
    }
    ```

### Task 3: `decltype` Inspection
Sometimes you need the exact type of a variable to declare another one, but you don't want to type it out.
*   **Task:** Create a variable `copy_data` that has the exact same type as `data` using `decltype(data)`.

## Common Pitfalls
*   **Accidental Copies:** `auto item : container` copies every element. Always use `auto&` or `const auto&` for non-primitive types.
*   **Dangling References:** `auto& x = get_temp_object();` is illegal/dangerous because the temporary object dies immediately.

## Verification
Compile and run the program. The output should verify that the data is iterated correctly.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
