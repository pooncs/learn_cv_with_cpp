# Exercise 08: std::atomic

## Goal
Use `std::atomic` to manage shared counters and flags without explicit mutex locking.

## Learning Objectives
1.  **Lock-Free Programming:** Understanding how hardware atomic instructions (CAS, fetch_add) work.
2.  **std::atomic:** Using atomic types (`std::atomic<int>`, `std::atomic<bool>`).
3.  **Memory Ordering:** (Bonus) Basic understanding of `memory_order_relaxed` vs `memory_order_seq_cst`.

## Practical Motivation
Locks are expensive (kernel trap). For simple counters (e.g., "processed frames count"), atomics are much faster and simpler.

## Step-by-Step Instructions

### Task 1: Atomic Counter
Create a global `std::atomic<int> counter{0}`.
Launch 10 threads, each incrementing it 10,000 times using `fetch_add` or `++`.
Verify result is 100,000.

### Task 2: Atomic Flag
Use `std::atomic<bool> stopFlag{false}` to signal threads to stop.
One thread waits `while(!stopFlag)`.
Another thread sets `stopFlag = true` after some time.

## Verification
1.  Run the counter test. It should be correct every time.
2.  Compare performance with `std::mutex` version (optional).

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
