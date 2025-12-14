# Exercise 02: Pixel Access Methods

## Goal
Benchmark `.at<>`, `ptr<>`, and iterator access methods in OpenCV.

## Learning Objectives
1.  Learn three ways to iterate over pixels in a `cv::Mat`.
2.  Understand the performance implications of each method.
3.  Write efficient image processing loops.

## Practical Motivation
Real-time CV requires high performance. 
- `at<>` includes bounds checking (slow, safe).
- `ptr<>` uses raw pointer arithmetic (fast, unsafe).
- Iterators provide STL compatibility (medium, safe).
Knowing when to use which is crucial.

## Theory & Background

### Access Methods
1.  **at<T>(y, x)**: Returns reference to element at row `y`, col `x`. Checks bounds in debug mode. Good for random access.
2.  **ptr<T>(y)**: Returns pointer to the beginning of row `y`. You can then access `row_ptr[x]`. Fastest sequential access.
3.  **MatIterator_<T>**: STL-style iterator. `it++`. Safer than pointers, slower than pointers.

## Implementation Tasks

### Task 1: Implement Access Functions
Implement functions that add 1 to every pixel using each method.

### Task 2: Benchmark
Measure execution time for a large image (e.g., 4K resolution) over multiple runs.

## Common Pitfalls
- Accessing `(x, y)` instead of `(y, x)` (Row, Col).
- Ignoring `step` (padding) when doing raw pointer arithmetic across rows (using `ptr(y)` avoids this).
