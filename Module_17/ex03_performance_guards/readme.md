# Exercise 03: Performance Regression Guards

## Goal
Implement a mechanism to detect and prevent performance regressions automatically. You will use Google Benchmark to measure code performance and a Python script to assert that it meets minimum throughput requirements.

## Learning Objectives
1.  **Benchmarking:** Use Google Benchmark to measure function throughput (items/s).
2.  **Data Export:** Export benchmark results to JSON format.
3.  **Regression Testing:** Analyze benchmark data programmatically to enforce Service Level Objectives (SLOs).
4.  **CI Integration:** Understand how to wire this into a build pipeline.

## Practical Motivation
In Computer Vision, "correct" code isn't enough; it must be fast. A developer might replace a raw loop with a fancy STL algorithm that allocates memory unnecessarily, dropping performance from 60 FPS to 10 FPS. Unit tests will pass, but the product is broken. A Performance Regression Guard catches this.

## Theory: Google Benchmark & JSON
Google Benchmark can output results to JSON:
```bash
./my_benchmark --benchmark_format=json --benchmark_out=results.json
```
The JSON contains fields like `cpu_time`, `real_time`, and custom counters like `items_per_second`.

## Step-by-Step Instructions

### Task 1: Create a Benchmark
Open `bench/bench_process.cpp`. Implement a benchmark for `process_image` (a simulated heavy function).
Use `state.SetItemsProcessed()` to track throughput.

```cpp
static void BM_ProcessImage(benchmark::State& state) {
    for (auto _ : state) {
        process_image(640, 480);
        state.SetItemsProcessed(1);
    }
}
BENCHMARK(BM_ProcessImage);
```

### Task 2: Build and Run
Configure CMake to link `benchmark::benchmark`.
Run the benchmark and inspect the output.

### Task 3: Performance Assertion Script
Create a Python script `tools/check_perf.py` that:
1.  Reads `results.json`.
2.  Finds the `BM_ProcessImage` result.
3.  Checks if `items_per_second` > 500 (simulated threshold).
4.  Exits with code 1 if the check fails.

### Task 4: Integrate
Run the full pipeline:
```bash
./build/bin/Release/bench_process --benchmark_format=json --benchmark_out=results.json
python tools/check_perf.py results.json 500
```

## Code Hints
-   **Python JSON Parsing:**
    ```python
    import json, sys
    data = json.load(open(sys.argv[1]))
    for b in data['benchmarks']:
        if b['name'] == 'BM_ProcessImage':
            if b['items_per_second'] < float(sys.argv[2]):
                sys.exit(1)
    ```

## Verification
1.  Run with the efficient implementation -> Check passes.
2.  Add a `std::this_thread::sleep_for` to the function -> Check fails.
