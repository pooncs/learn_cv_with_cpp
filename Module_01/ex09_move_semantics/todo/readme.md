# Exercise 09: Move Semantics

## Goal
Optimize a heavy `Matrix` class to support **Move Semantics**. You will see how "stealing" resources is much faster than copying them.

## Learning Objectives
1.  Understand **L-values** (Named objects) vs **R-values** (Temporary objects).
2.  Implement a **Move Constructor** and **Move Assignment Operator**.
3.  Use `std::move` to cast an L-value to an R-value.
4.  Follow the **Rule of Five**.

## Analogy: The House Move
*   **Copy (Deep Copy):** You want to move to a new house.
    *   You buy brand new furniture identical to your old stuff.
    *   You put it in the new house.
    *   You destroy the old house with the old furniture inside.
    *   *Result:* Huge waste of money and time.
*   **Move (Shallow Copy / Steal):**
    *   You take the furniture from the old house.
    *   You put it in the new house.
    *   The old house is now empty.
    *   *Result:* Fast and efficient.

## Practical Motivation
In CV, we handle large images (e.g., 4K resolution, 50MB).
Passing `cv::Mat` by value *used* to be expensive if it triggered a deep copy. Modern `cv::Mat` uses reference counting (like `shared_ptr`), but if you write your own data structures (e.g., a Point Cloud buffer), you **must** implement move semantics to avoid unnecessary copies when returning from functions or resizing vectors.

## Step-by-Step Instructions

### Task 1: The Heavy Class
Open `src/main.cpp`. The `Matrix` class manages a raw pointer `int* data`.
*   It currently has a **Copy Constructor** (Expensive loop).
*   It lacks Move operations.

### Task 2: Implement Move Constructor
Add `Matrix(Matrix&& other) noexcept`.
1.  **Steal:** specific `this->data = other.data`.
2.  **Nullify:** `other.data = nullptr`. (Crucial! Otherwise destructor will double-free).

### Task 3: Implement Move Assignment
Add `Matrix& operator=(Matrix&& other) noexcept`.
1.  **Check Self-Assignment:** `if (this != &other)`.
2.  **Clean Up:** `delete[] data` (Free current resource).
3.  **Steal & Nullify:** Copy pointer, set other to null.

### Task 4: Benchmark
In `main`, we create a `vector<Matrix>` and push back a temporary.
*   Compare the number of "Copy" prints vs "Move" prints.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show "Move Constructor" being called instead of "Copy Constructor" when pushing back temporary objects.
