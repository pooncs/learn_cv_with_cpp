# Exercise 10: RAII Wrapper (The Rule of Five)

## Goal
Combine everything learned in Module 01 to build a production-quality `ImageBuffer` class that manages memory automatically using **RAII** and implements the **Rule of Five**.

## Learning Objectives
1.  **RAII (Resource Acquisition Is Initialization):** Constructor allocates, Destructor frees.
2.  **Rule of Five:** If you implement one, you likely need all five:
    *   Destructor
    *   Copy Constructor
    *   Copy Assignment
    *   Move Constructor
    *   Move Assignment
3.  Self-assignment checks.

## Practical Motivation
This is the core of how `cv::Mat` or `std::vector` works internally. Writing this yourself gives you deep insight into memory management bugs (double frees, leaks) that plague C++ CV applications.

## Theory
*   **Deep Copy:** Allocate new memory, copy values (`memcpy`). Used when we want two independent images.
*   **Move:** Steal the pointer. Used when passing images through a pipeline.

## Step-by-Step Instructions

### Task 1: Scaffolding
Open `src/ImageBuffer.cpp` and `include/ImageBuffer.hpp`. The Constructor and Destructor are already there (partially).

### Task 2: Copy Constructor
`ImageBuffer(const ImageBuffer& other)`
1.  Allocate new `data` of size `width * height`.
2.  `std::memcpy` from `other.data`.

### Task 3: Copy Assignment Operator
`ImageBuffer& operator=(const ImageBuffer& other)`
1.  **Check for self-assignment:** `if (this == &other) return *this;` (CRITICAL!)
2.  `delete[]` old data (prevent leak).
3.  Allocate new data and copy.
4.  Return `*this`.

### Task 4: Move Operations
Implement Move Constructor and Move Assignment (similar to Exercise 09).
*   Remember to `delete[]` in Move Assignment before stealing!

## Common Pitfalls
*   **Forgetting `delete[]` in Assignment:** Leak!
*   **Forgetting Self-Assignment Check:** `a = a` deletes `a`'s data, then tries to copy from deleted memory. Crash!
*   **Shallow Copy:** Just copying the pointer `data = other.data`. Both destructors will try to free the same memory. Double Free Crash!

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
The provided `main.cpp` tests copy and move scenarios. Ensure no crashes occur.
