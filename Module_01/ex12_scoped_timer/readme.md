# Exercise 12: Scoped Timer

## Goal
Implement RAII timing with `std::chrono` for profiling.

## Learning Objectives
1.  Understand RAII (Resource Acquisition Is Initialization) for timing.
2.  Use `std::chrono` for high-resolution timing.
3.  Automatically log duration when a scope exits.

## Practical Motivation
Profiling code is essential for optimization. Manually calling `start()` and `stop()` is error-prone and tedious. A scoped timer handles this automatically.

## Theory & Background

### std::chrono
- `std::chrono::high_resolution_clock`: The clock with the shortest tick period.
- `std::chrono::time_point`: Represents a point in time.
- `std::chrono::duration`: Represents a time interval.

### RAII
Constructor acquires resource (start time).
Destructor releases resource (calculates duration and prints).

## Implementation Tasks

### Task 1: ScopedTimer Class
Create a class `ScopedTimer` that:
- Captures start time in constructor.
- Captures end time in destructor.
- Prints the elapsed time to stdout.

### Task 2: Usage
Use the timer to measure the execution time of a slow function (e.g., sorting a large vector).

## Common Pitfalls
- Using `system_clock` (wall clock) instead of `steady_clock` or `high_resolution_clock` for intervals.
