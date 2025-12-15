# Exercise 02: Data Races

## Goal
Demonstrate a data race and fix it using `std::mutex`.

## Learning Objectives
1.  **Data Race:** Understand what happens when two threads write to the same memory location concurrently without synchronization.
2.  **Mutex:** Use `std::mutex` to protect critical sections.
3.  **Performance Cost:** Observe that locking slows down parallel execution compared to unsynchronized (but wrong) execution.

## Practical Motivation
In CV, if multiple threads update a shared global histogram or a counter, the final result will be wrong due to lost updates.

## Theory
`counter++` is not atomic. It involves: Read, Increment, Write.
Thread A reads 5. Thread B reads 5. Both increment to 6 and write. Result is 6, but should be 7.

## Step-by-Step Instructions

### Task 1: Unsafe Counter
Implement a class `UnsafeCounter` with an `increment()` method.
Launch 10 threads, each calling `increment()` 1000 times.
Check if the final count is 10000. It likely won't be.

### Task 2: Safe Counter
Implement `SafeCounter` using `std::mutex`.
Wrap the increment logic in `mutex.lock()` and `mutex.unlock()` (or better, `std::lock_guard`).

## Verification
1.  Run the unsafe version multiple times. It should fail (result < expected).
2.  Run the safe version. It should always pass.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
