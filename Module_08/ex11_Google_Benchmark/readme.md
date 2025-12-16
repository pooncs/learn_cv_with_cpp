# Module 08 - Exercise 11: Google Benchmark Integration

## Goal
Integrate Google Benchmark into a CMake project using Conan/FetchContent and run a simple microbenchmark.

## Concept
**Google Benchmark** is a library to measure the performance of C++ code snippets. It handles:
- Running code repeatedly to get statistically significant results.
- Calculating mean, median, and standard deviation.
- Outputting results to JSON/CSV.

## Task
1.  Add `benchmark` to dependencies.
2.  Write a simple benchmark that measures `std::string` creation.
3.  Link against `benchmark::benchmark` and `benchmark::benchmark_main`.

## Instructions
1.  Navigate to `todo/`.
2.  Inspect `conanfile.txt` (ensure benchmark is listed).
3.  Edit `CMakeLists.txt` to find and link benchmark.
4.  Edit `src/main.cpp` to define a benchmark function.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build . --config Release
# Run it!
./bin/bench_demo
```
