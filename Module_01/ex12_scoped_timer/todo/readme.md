# Exercise 12: Scoped Timer (Profiling)

## Goal
Implement a `ScopedTimer` class using RAII and `std::chrono` to automatically measure and print the execution time of a block of code.

## Learning Objectives
1.  **RAII for Timing:** Start time in constructor, End time in destructor.
2.  **std::chrono:** High-resolution clock, time points, and durations.
3.  **Scoped Profiling:** Measure specific blocks `{ ... }` easily.

## Analogy: The Stopwatch Coach
*   **Old C++ (Manual Timing):**
    *   `auto start = clock();`
    *   Do work.
    *   `auto end = clock();`
    *   `cout << end - start;`
    *   *Problem:* Tedious. If you have 10 exit points, you need 10 print statements.
*   **Modern C++ (Scoped Timer):**
    *   You hire a **Stopwatch Coach**.
    *   When you enter the room (Scope Start), the Coach clicks "Start".
    *   When you leave the room (Scope End), the Coach clicks "Stop" and yells your time.
    *   It doesn't matter *how* you leave (Return, Exception, Walk out) - the Coach always stops the clock.

## Practical Motivation
Performance optimization is key in CV. You need to know:
*   "How long does `cv::GaussianBlur` take?"
*   "Is my loop slow?"
*   "Does loading the image take longer than processing it?"

`ScopedTimer` lets you sprinkle `{ ScopedTimer t("Label"); ... }` blocks around your code to get instant feedback.

## Step-by-Step Instructions

### Task 1: The Timer Class
Open `src/main.cpp`. Create `class ScopedTimer`.
*   Member: `std::string name`, `time_point start`.
*   Constructor: Capture `name`, set `start = std::chrono::high_resolution_clock::now()`.
*   Destructor:
    *   Get `end` time.
    *   Calculate duration (`end - start`).
    *   Print `[name] took X ms`.

### Task 2: Profile Code
In `main()`:
1.  Create a scope `{ ... }`.
2.  Instantiate `ScopedTimer t("Heavy Work");`.
3.  Simulate work (e.g., `std::this_thread::sleep_for`).
4.  Observe the output when the scope ends.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show the elapsed time for the block.
