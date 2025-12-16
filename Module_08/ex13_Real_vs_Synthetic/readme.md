# Module 08 - Exercise 13: Real vs Synthetic Data

## Goal
Understand the importance of benchmarking with representative data.

## Task
Benchmark a simple image processing operation (e.g., brightness adjustment) on:
1.  **Synthetic Data**: A perfectly random noise image.
2.  **Real Data**: A structured image (gradient or loaded file - we'll simulate a gradient here).

## Instructions
1.  Implement `BM_ProcessRandom` (using `rand()`).
2.  Implement `BM_ProcessGradient` (structured data).
3.  Compare if branch prediction or cache effects make structured data faster.

## Build
```bash
mkdir build
cd build
conan install .. --build=missing
cmake .. --preset conan-default
cmake --build . --config Release
./bin/bench_real_syn
```
