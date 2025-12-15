# Exercise 01: std::thread and Parallel Image Processing

## Goal
Learn how to use `std::thread` to parallelize image processing tasks.

## Learning Objectives
1.  **Thread Creation:** Launching threads with `std::thread`.
2.  **Joining:** Waiting for threads to finish using `.join()`.
3.  **Data Partitioning:** Splitting an image into blocks (strips) to process them independently.

## Practical Motivation
Image processing operations (like brightness adjustment, thresholding, or convolution) are often "embarrassingly parallel". Splitting the work across CPU cores can linearly speed up the pipeline.

## Theory
`std::thread t(function, args...);` starts a new thread.
`t.join()` blocks the calling thread until `t` finishes.
To process an image in parallel, we divide the rows among threads.
Thread $i$ processes rows from $\frac{H}{N} \cdot i$ to $\frac{H}{N} \cdot (i+1)$.

## Step-by-Step Instructions

### Task 1: Single-Threaded Reference
Implement a function `processImage(img)` that inverts the colors of the image sequentially. Measure its time.

### Task 2: Multi-Threaded Implementation
Implement `processImageParallel(img, numThreads)`.
1.  Calculate start and end rows for each thread.
2.  Launch `std::thread` for each strip.
3.  Store threads in a `std::vector<std::thread>`.
4.  Join all threads.

### Task 3: Benchmarking
Compare the execution time of single-threaded vs multi-threaded (e.g., 4 threads) on a large image.

## Common Pitfalls
1.  **Dangling References:** Ensure the image stays alive while threads are running. Passing by reference `std::ref` or pointer is common.
2.  **False Sharing:** If threads write to adjacent memory locations on the same cache line, performance drops. Processing distinct rows usually avoids this.
3.  **Oversubscription:** Launching more threads than hardware cores causes context switching overhead. Use `std::thread::hardware_concurrency()`.

## Verification
1.  Check that the output image is identical to the single-threaded version.
2.  Observe speedup (for large images).

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
