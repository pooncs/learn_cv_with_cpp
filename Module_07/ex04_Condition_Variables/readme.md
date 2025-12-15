# Exercise 04: Condition Variables and Thread-Safe Queue

## Goal
Implement a thread-safe queue using `std::condition_variable` to synchronize producer and consumer threads.

## Learning Objectives
1.  **Condition Variables:** `std::condition_variable` allows a thread to sleep until a condition is met (signaled by another thread).
2.  **Spurious Wakeups:** Understanding why we use `while (!condition) wait(...)` instead of `if`.
3.  **Thread-Safe Queue:** A fundamental data structure for pipelines.

## Practical Motivation
In a CV pipeline, the camera thread pushes frames into a buffer, and the processing thread pops them. If the buffer is empty, the processor should sleep (not busy-wait). If the buffer is full, the camera should drop frames or wait.

## Theory
`cv.wait(lock, predicate)` atomically unlocks the mutex and sleeps. When notified, it re-locks and checks the predicate.

## Step-by-Step Instructions

### Task 1: SafeQueue Class
Implement a template class `SafeQueue<T>`.
-   `push(T item)`: Lock mutex, push to `std::queue`, notify one.
-   `pop(T& item)`: Lock mutex, wait until not empty, pop, return.

### Task 2: Test with Threads
Launch a producer pushing integers and a consumer popping them. Verify order and count.

## Common Pitfalls
1.  **Lost Wakeups:** If you notify before the waiter is waiting, it's fine (predicate check handles it). But if you forget to notify, the waiter hangs forever.
2.  **Deadlock:** Calling `notify` while holding the lock is allowed but can cause "hurry up and wait". Unlocking before notifying is an optimization.

## Verification
1.  Run the test. Ensure no items are lost and the program terminates.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
