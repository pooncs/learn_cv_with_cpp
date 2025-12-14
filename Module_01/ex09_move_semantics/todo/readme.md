# Exercise 09: Move Semantics (std::move)

## Goal
Optimize a heavy `Matrix` class by implementing **Move Semantics**. Understand how to transfer ownership of resources (pointers) instead of copying them.

## Learning Objectives
1.  L-values vs R-values (basics).
2.  Implement a **Move Constructor**.
3.  Implement a **Move Assignment Operator**.
4.  Use `std::exchange` for cleaner pointer swapping.

## Practical Motivation
Imagine you have a function `get_image()` that returns a 4K image (24MB).
*   **Copy:** `Image A = get_image();` -> Allocates new 24MB, copies data, deletes old 24MB. Slow!
*   **Move:** `Image A = get_image();` -> A just takes the pointer from the temporary result. Zero allocation. Instant.

## Theory
*   **L-value:** Something with a name (e.g., variable `x`).
*   **R-value:** A temporary (e.g., `get_image()`, `5`, `std::move(x)`).
*   **Move Constructor:** Takes `Type&& other`. You steal `other`'s resources and leave `other` in a valid but empty state (e.g., pointer = nullptr).

## Step-by-Step Instructions

### Task 1: Implement Move Constructor
Open `src/main.cpp`. The `BigMatrix` class manages a raw `float* data`.
*   Signature: `BigMatrix(BigMatrix&& other) noexcept`
*   **Action:**
    1.  Copy `other.data` pointer to `this->data`.
    2.  Set `other.data` to `nullptr` (so its destructor doesn't free the memory we just stole).
    3.  Copy `other.size`.
    4.  Set `other.size` to 0.

### Task 2: Use `std::move`
In `main()`:
1.  Create `BigMatrix m1(1000)`.
2.  Move it to `m2`: `BigMatrix m2 = std::move(m1);`.
3.  Verify that `m1` is now empty (pointer is null) and `m2` has the data.

## Common Pitfalls
*   **Using a Moved-from Object:** `std::cout << m1.data[0]` after move is Undefined Behavior (usually a crash).
*   **Not marking `noexcept`:** If your move constructor isn't `noexcept`, `std::vector` might refuse to use it during resizing (it will copy instead).

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should confirm "Moved" and verify m1 is empty.
