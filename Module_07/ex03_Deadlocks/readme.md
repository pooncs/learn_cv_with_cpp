# Exercise 03: Deadlocks

## Goal
Simulate a deadlock and fix it using `std::lock` (C++11) or `std::scoped_lock` (C++17).

## Learning Objectives
1.  **Deadlock:** A situation where Thread A holds Lock 1 and waits for Lock 2, while Thread B holds Lock 2 and waits for Lock 1.
2.  **Lock Ordering:** A common strategy to avoid deadlocks (always acquire locks in the same order).
3.  **Standard Library Tools:** `std::lock(m1, m2)` uses a deadlock-avoidance algorithm to lock multiple mutexes safely.

## Practical Motivation
In a multi-threaded CV pipeline, you might have a `Camera` object and a `Display` object, both with mutexes. If one thread tries to `grab()` (lock cam, then display) and another tries to `resize()` (lock display, then cam), you get a freeze.

## Theory
`std::scoped_lock` (C++17) is a RAII wrapper that can take multiple mutexes and lock them safely using the same algorithm as `std::lock`.

## Step-by-Step Instructions

### Task 1: Simulate Deadlock
Create two resources `r1`, `r2` (mutexes).
Thread A: locks `r1`, sleeps 1ms, locks `r2`.
Thread B: locks `r2`, sleeps 1ms, locks `r1`.
Run this. The program should hang.

### Task 2: Fix with std::lock
Refactor the locking logic to use `std::lock(r1, r2)`.
Or use `std::scoped_lock lock(r1, r2)`.

## Verification
1.  Run the deadlock simulation (it should hang).
2.  Run the fix (it should complete).

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
