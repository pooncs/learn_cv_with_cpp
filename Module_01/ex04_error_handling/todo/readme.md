# Exercise 04: Modern Error Handling (std::optional)

## Goal
Replace error codes and "magic values" (like returning `-1` for failure) with **`std::optional`**, making function signatures self-documenting and type-safe.

## Learning Objectives
1.  Understand why returning magic numbers (e.g., `NaN`, `-1`) is bad.
2.  Use `std::optional<T>` to represent a value that might be missing.
3.  Access values safely using `.has_value()` and `.value()`.
4.  Use `.value_or()` for default fallbacks.

## Practical Motivation
In CV, many operations can fail silently:
*   Finding a chessboard in an image.
*   Calculating the intersection of parallel lines.
*   Reading a file that doesn't exist.

Returning a `bool` (success/fail) forces you to pass the result by reference (`bool solve(Input, Output&)`) which is ugly. Returning `-1` is ambiguous (what if the result *is* -1?).
`std::optional` explicitly states: "I might return a value, or I might return nothing."

## Theory
`std::optional<T>` is a stack-allocated wrapper that contains storage for `T` and a boolean flag.
*   **Size:** `sizeof(T) + alignment_padding + bool`.
*   **No Allocation:** Unlike `unique_ptr`, it does not allocate memory on the heap.

## Step-by-Step Instructions

### Task 1: Refactor Function Signature
Open `src/main.cpp`. The function `safe_sqrt(double x)` currently returns `-1.0` if `x < 0`.
*   **Task:** Change return type to `std::optional<double>`.
*   **Task:** Return `std::nullopt` (or `{}`) on failure.
*   **Task:** Return the result directly on success (implicit conversion).

### Task 2: Handle the Result
In `main()`:
1.  Call `safe_sqrt`.
2.  Check if it succeeded using `if (result.has_value())` or simply `if (result)`.
3.  Print the value using `*result` or `result.value()`.

### Task 3: Default Value
Use `.value_or(0.0)` to print a default value if the calculation failed, without writing an `if` statement.

## Common Pitfalls
*   **Unchecked Access:** Calling `.value()` or `*opt` on an empty optional throws `std::bad_optional_access` (or undefined behavior for `*`). Always check first.
*   **Performance:** Passing `std::optional<LargeObject>` by value copies the object. Use `std::optional<std::reference_wrapper<T>>` or pointers if copying is expensive (though optional references are not standard in C++17, pointers are preferred there). For primitives (`double`, `int`), optional is perfect.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
The output should handle the negative input gracefully without printing `-1`.
