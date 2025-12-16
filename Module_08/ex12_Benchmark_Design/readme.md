# Module 08 - Exercise 12: Benchmark Design

## Goal
Design reproducible microbenchmarks using arguments and complexity analysis.

## Concept
Benchmarking isn't just about running once. We often want to see how performance scales with input size (O(N) vs O(N^2)).
Google Benchmark supports `->Range(start, end)` to automatically sweep input sizes.

## Task
Benchmark `std::vector::push_back` with and without `reserve`.
1.  Define a benchmark that takes an argument (size).
2.  Run loop for `state.range(0)` iterations.
3.  Compare "With Reserve" vs "Without Reserve".

## Instructions
1.  Navigate to `todo/`.
2.  Implement `BM_VectorPush` and `BM_VectorReserve`.
3.  Use `BENCHMARK(BM_...)->Range(8, 8<<10);` to sweep sizes from 8 to 8192.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build . --config Release
./bin/bench_design
```
