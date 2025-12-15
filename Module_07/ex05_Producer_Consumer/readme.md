# Exercise 05: Producer-Consumer Pipeline

## Goal
Build a realistic processing pipeline where one thread simulates capturing frames (Producer) and another processes them (Consumer).

## Learning Objectives
1.  **Pipeline Architecture:** Decoupling acquisition and processing.
2.  **Latency vs Throughput:** Understanding how buffering helps smooth out jitter.
3.  **Graceful Shutdown:** Stopping the pipeline cleanly.

## Practical Motivation
In a real camera app, the camera driver delivers frames at 30 FPS. If processing a frame takes 40ms (>33ms), we might drop frames or lag. A buffer helps if the processing time varies (jitter).

## Step-by-Step Instructions

### Task 1: Frame Structure
Define a simple `Frame` struct (id, timestamp, data).

### Task 2: Pipeline Class
Use the `SafeQueue` from Ex 04.
-   `start()`: Launches producer and consumer threads.
-   `stop()`: Signals threads to stop and joins them.

### Task 3: Producer Logic
Loop:
1.  Create a frame (increment ID).
2.  Sleep for 33ms (simulating 30 FPS).
3.  Push to queue.

### Task 4: Consumer Logic
Loop:
1.  Pop frame.
2.  Process it (simulate work with sleep, e.g., 20-50ms random).
3.  Log "Processed Frame X".

## Verification
1.  Run the pipeline for 5 seconds.
2.  Observe the logs.
3.  Ensure all threads exit when `stop()` is called.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
