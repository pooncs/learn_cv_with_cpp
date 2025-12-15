# Exercise 06: std::async and Futures

## Goal
Use `std::async` to run independent tasks in parallel and retrieve their results using `std::future`.

## Learning Objectives
1.  **High-Level Concurrency:** `std::async` is simpler than `std::thread` for tasks that return a value.
2.  **Futures:** `std::future<T>` represents a value that will be available later.
3.  **Synchronization:** `.get()` blocks until the result is ready, handling synchronization automatically.

## Practical Motivation
Imagine a surveillance system that needs to:
1.  Detect faces (slow).
2.  Compute color histogram (fast).
3.  Save frame to disk (I/O bound).
These tasks are independent. We can launch them all at once and wait for them to finish before processing the next frame.

## Step-by-Step Instructions

### Task 1: Define Tasks
Create 3 dummy functions:
-   `detectFaces()`: Sleeps 50ms, returns int (count).
-   `computeHistogram()`: Sleeps 20ms, returns float (mean).
-   `saveToDisk()`: Sleeps 80ms, returns bool (success).

### Task 2: Launch Async
Use `std::async(std::launch::async, ...)` to start them.

### Task 3: Collect Results
Call `.get()` on the futures to retrieve values.
Measure total time. It should be roughly `max(50, 20, 80)` instead of `50+20+80`.

## Common Pitfalls
1.  **Destructor Blocking:** If you don't assign the return value of `std::async` to a variable (a `std::future`), the destructor of the temporary future will block immediately! This makes it sequential.
2.  **Launch Policy:** `std::launch::deferred` means it runs only when `.get()` is called (lazy evaluation, sequential). Always use `std::launch::async` for parallelism.

## Verification
1.  Run the program.
2.  Check that total time is significantly less than the sum of individual times.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
