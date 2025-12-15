# Exercise 06: Code Coverage

## Goal
Measure how much of your code is executed by your test suite using `gcov` (GCC/Clang) or OpenCppCoverage (Windows).

## Learning Objectives
1.  **Coverage Metrics:** Understand line coverage, function coverage, and branch coverage.
2.  **Instrumentation:** Compile code with coverage tracking enabled.
3.  **Reporting:** Generate human-readable HTML reports.
4.  **Gap Analysis:** Identify untested code paths.

## Practical Motivation
If you have 100 tests, do they cover 10% or 90% of your code? Coverage reports tell you exactly which lines were never executed, highlighting risky areas that need more tests.

## Step-by-Step Instructions

### Task 1: Enable Coverage Flags (GCC/Clang)
```cmake
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()
```
For MSVC, it's more complex, often requiring external tools like OpenCppCoverage.

### Task 2: Run Tests
Run your test executable (e.g., `ctest`). This generates `.gcda` files.

### Task 3: Generate Report
Use `lcov` and `genhtml`:
```bash
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory out
```
Open `out/index.html` in a browser.

### Task 4: Analyze
Find the function in `src/algo.cpp` that has a specific `if` branch that is red (not executed). Write a test case to cover it.

## Verification
The generated report should show 100% line coverage after adding the missing test.
