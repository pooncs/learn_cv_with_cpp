# Exercise 11: Thread-Safe Buffer Pool

## Goal
Implement a `BufferPool` that manages a fixed set of reusable `cv::Mat` objects (or raw byte buffers) to avoid expensive memory allocations in a high-speed pipeline.

## Learning Objectives
1.  **Memory Pooling:** Reusing memory blocks instead of `new`/`delete` (or `malloc`/`free`) reduces heap fragmentation and OS overhead.
2.  **RAII for Buffers:** Using a custom smart pointer or handle (`BufferHandle`) that automatically returns the buffer to the pool when it goes out of scope.
3.  **Synchronization:** Protecting the pool with mutexes and condition variables.

## Practical Motivation
In a 4K 60FPS pipeline, allocating 24MB buffers 60 times a second puts huge pressure on the memory allocator. A pool of 10 pre-allocated buffers is much more efficient.

## Step-by-Step Instructions

### Task 1: BufferPool Class
-   Members: `std::stack<cv::Mat> pool`, `std::mutex`, `std::condition_variable`.
-   Constructor: Pre-allocate $N$ buffers of fixed size.
-   `acquire()`: Pop a buffer from the stack. If empty, block until one is released.
-   `release(cv::Mat)`: Push buffer back and notify.

### Task 2: BufferHandle (Optional but recommended)
-   Create a class `BufferHandle` that holds a `cv::Mat` and a reference to the pool.
-   Destructor calls `pool.release(buffer)`.

### Task 3: Test
-   Launch multiple threads that `acquire()` a buffer, fill it, and `release()` it.
-   Verify that no more than $N$ buffers are in use simultaneously.

## Common Pitfalls
1.  **Buffer Size:** Ensure reused buffers are the correct size/type. If the user needs a different size, re-allocation might be needed (or throw error).
2.  **Exception Safety:** If `acquire` throws, ensure state remains consistent.

## Verification
1.  Create a pool of size 2.
2.  Launch 4 threads.
3.  Ensure threads block and wait for buffers, and all eventually complete.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
