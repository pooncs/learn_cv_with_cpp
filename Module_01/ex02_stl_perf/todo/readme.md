# Exercise 02: STL Containers Performance

## Goal
Understand the performance characteristics of `std::vector` vs. `std::list` (and `std::deque`) to make informed decisions when designing high-performance CV pipelines.

## Learning Objectives
1.  Understand **Contiguous Memory** vs. **Linked Nodes**.
2.  Benchmark **Insertion** vs. **Random Access** vs. **Traversal**.
3.  Learn why `std::vector` is almost always the default choice in Computer Vision.

## Practical Motivation
In CV, an image is a contiguous block of memory (`width * height * channels`). Accessing pixels row-by-row is fast because the CPU prefetcher loads the next chunk of memory into the cache before you ask for it.
*   **Vector:** Like an image row. Fast.
*   **List:** Like scattered pixels linked by pointers. Slow due to cache misses.

If you choose `std::list` to store 1 million feature points, your algorithm might run 10x slower than with `std::vector` solely due to memory latency.

## Theory: Memory Layout

### `std::vector`
*   **Layout:** `[ 1 ][ 2 ][ 3 ][ 4 ]` (One solid block).
*   **Access:** O(1) math. `ptr + index`.
*   **Insert at End:** O(1) amortized.
*   **Insert in Middle:** O(N) (Everything must shift).

### `std::list` (Doubly Linked List)
*   **Layout:** `[Prev| 1 |Next] <---> [Prev| 2 |Next] ...` (Scattered in heap).
*   **Access:** O(N) (Must hop from node to node).
*   **Insert at End:** O(1).
*   **Insert in Middle:** O(1) (Just rewire pointers).

## Step-by-Step Instructions

### Task 1: Setup Benchmarking
Open `src/main.cpp`. We will use `std::chrono` to measure time.
*   Define `N = 100,000` elements.

### Task 2: Benchmark `push_back`
1.  Start a timer.
2.  Loop `N` times and `push_back(i)` into a `std::vector`.
3.  Stop timer and print result.
4.  Repeat for `std::list`.
*   *Hypothesis:* Vector might be slightly faster or similar, but list allocates memory for *every* node, which is slow.

### Task 3: Benchmark Random Access
1.  Try to sum all elements by accessing `container[i]`.
2.  **Vector:** `sum += vec[i]`.
3.  **List:** You **cannot** use `[]`. You must use an iterator or `std::advance` (which is O(N)).
    *   *Note:* Do not implement O(N^2) list access. Instead, simulate a scenario where you *need* the 50,000th element.

### Task 4: Benchmark Linear Traversal
1.  Iterate from `begin()` to `end()` and sum values.
2.  Compare Vector vs. List.
*   *Hypothesis:* Vector is significantly faster due to CPU caching (Spatial Locality).

## Common Pitfalls
*   **Premature Optimization:** "I need to insert in the middle, so I'll use List."
    *   *Reality:* Unless the object is huge or N is massive, shifting a vector of ints is often faster than traversing a list to find the insertion point.
*   **Pointer Invalidation:** Vector reallocations invalidate pointers. List does not.

## Verification
Run the code.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
**Expected Output:** Vector traversal should be drastically faster (microsecond scale) than List traversal.
