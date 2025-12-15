# Exercise 09: C++17 Parallel Algorithms

## Goal
Use C++17 parallel execution policies (`std::execution::par`) to parallelize standard algorithms like `std::for_each`, `std::transform`, and `std::sort`.

## Learning Objectives
1.  **Execution Policies:** Understand `seq`, `par`, and `par_unseq`.
2.  **Simplicity:** Parallelizing code by just adding an argument to standard algorithms.
3.  **Performance:** Benchmarking against sequential execution.

## Practical Motivation
Before C++17, you had to write `std::thread` or OpenMP code manually. Now, you can simply tell the STL to run in parallel. This is great for pixel-wise operations on large arrays (e.g., LUT application, gamma correction).

## Theory
-   `std::execution::seq`: Sequential execution (default).
-   `std::execution::par`: Parallel execution. Tasks run on multiple threads. Order is indeterminate.
-   `std::execution::par_unseq`: Parallel and Vectorized (SIMD).

## Step-by-Step Instructions

### Task 1: Large Data Setup
Create a `std::vector<double>` with 10 million elements.

### Task 2: Compute Heavy Operation
Define a lambda that performs a heavy math operation (e.g., `sin(x) * cos(x) + sqrt(x)`).

### Task 3: Benchmark
1.  Run `std::transform(std::execution::seq, ...)` and measure time.
2.  Run `std::transform(std::execution::par, ...)` and measure time.

## Common Pitfalls
1.  **Thread Safety:** The operation applied to each element must be thread-safe (no side effects on shared variables without synchronization).
2.  **Overhead:** For small vectors, parallelization overhead might make it slower.
3.  **Compiler Support:** Requires C++17. On MSVC, it uses PPL. On GCC/Clang, it often requires Intel TBB.

## Verification
1.  Observe speedup on large datasets.
2.  Verify results are correct.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
