# Exercise 08: Profiling

## Goal
Identify performance bottlenecks using profiling tools.

## Learning Objectives
1.  **Hotspots:** Functions that consume the most CPU time.
2.  **Instrumentation:** Using tools like `perf` (Linux) or Visual Studio Profiler (Windows).
3.  **Flame Graphs:** Visualizing call stacks.

## Practical Motivation
"Premature optimization is the root of all evil." Don't guess where code is slow. Measure it.

## Step-by-Step Instructions
1.  Write a program with a deliberately slow function (e.g., inefficient sort or heavy math).
2.  Run the profiler.
3.  Identify the slow function.
4.  Optimize it.
5.  Profile again to confirm speedup.

## Verification
*   Screenshot or text report from the profiler showing the bottleneck.
