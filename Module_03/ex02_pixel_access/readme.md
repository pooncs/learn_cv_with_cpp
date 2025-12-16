# Exercise 02: Pixel Access Methods

## Goal
Benchmark `.at<>`, `ptr<>`, and iterator access methods in OpenCV.

## Learning Objectives
1.  **Three Ways:** Learn `.at()`, `.ptr()`, and `Iterator`.
2.  **Performance:** Why pointers are 10x faster than `.at()`.
3.  **Safety:** Why `.at()` crashes nicely, but pointers crash horribly.

## Analogy: The Delivery Driver
You need to deliver packages to every house in a city (Every pixel in an image).
*   **`.at<uchar>(y, x)` (Random Access):**
    *   The driver stops at every house.
    *   Checks GPS: "Am I at the right address? Is this house 500?"
    *   Rings doorbell.
    *   *Result:* Very safe, but incredibly slow.
*   **`.ptr<uchar>(y)` (Raw Pointer):**
    *   The driver goes to the start of the street (Row).
    *   He knows the houses are exactly 10 meters apart.
    *   He drives down the street at full speed, throwing packages out the window.
    *   *Result:* Extremely fast. But if the street ends and he keeps throwing... he hits a pedestrian (Segfault).
*   **`MatIterator`:**
    *   The driver follows a strict, pre-planned route.
    *   *Result:* Good balance, standard C++ style.

## Practical Motivation
Real-time CV (30 FPS) gives you 33ms per frame.
If you process a 4K image using `.at()`, it might take 100ms. Using pointers, it might take 5ms.
*   **Use `.at()`** for debugging or accessing single pixels (e.g., clicking on an image).
*   **Use `.ptr()`** for image processing loops (filters, color conversion).

## Step-by-Step Instructions

### Task 1: The Safe Way (`at`)
Open `src/main.cpp`.
*   Implement `process_at(Mat& img)`.
*   Loop `y` from 0 to rows, `x` from 0 to cols.
*   Increment pixel: `img.at<uchar>(y, x) += 1;`.

### Task 2: The Fast Way (`ptr`)
*   Implement `process_ptr(Mat& img)`.
*   Loop `y` from 0 to rows.
*   Get row pointer: `uchar* row_ptr = img.ptr<uchar>(y);`.
*   Loop `x` from 0 to cols.
*   Increment: `row_ptr[x] += 1;`.

### Task 3: The Iterator Way
*   Implement `process_iter(Mat& img)`.
*   Use `cv::MatIterator_<uchar> it`.
*   Loop from `begin` to `end`.
*   Increment: `(*it) += 1;`.

### Task 4: Benchmark
*   We use `ScopedTimer` (from Module 1) to measure each function on a large image ($2000 \times 2000$).
*   Run the program and compare times.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Expected: `ptr` should be fastest, followed by `iterator`, with `at` being significantly slower (in Debug mode). In Release mode, `at` might be optimized, but `ptr` remains king.
