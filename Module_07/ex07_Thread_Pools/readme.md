# Exercise 07: Thread Pools

## Goal
Implement a fixed-size Thread Pool to manage a set of worker threads and a task queue.

## Learning Objectives
1.  **Thread Management:** Avoiding the overhead of creating/destroying threads for every small task.
2.  **Task Queue:** Using a thread-safe queue to distribute work.
3.  **Functional Programming:** Using `std::function` and lambdas to pass tasks.

## Practical Motivation
Launching a new thread costs OS resources (stack memory, kernel structures). If we have 1000 tiny tasks (e.g., processing 1000 image patches), creating 1000 threads is inefficient. A thread pool with $N$ workers (where $N \approx$ CPU cores) is much better.

## Step-by-Step Instructions

### Task 1: ThreadPool Class
-   `std::vector<std::thread> workers`
-   `SafeQueue<std::function<void()>> tasks`
-   Constructor: Launch $N$ workers.
-   Worker Loop: Pop task, execute it.

### Task 2: Submission
-   `enqueue(task)`: Push task to queue.

### Task 3: Graceful Shutdown
-   Destructor: Signal stop, join all threads.

## Common Pitfalls
1.  **Infinite Waiting:** If the queue is empty, workers sleep. You must notify them to wake up and exit when shutting down.
2.  **Exception Handling:** If a task throws, the worker thread might crash. Wrap execution in try-catch.

## Verification
1.  Submit 100 tasks that print their ID.
2.  Verify all run.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
