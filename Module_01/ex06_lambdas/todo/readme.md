# Exercise 06: Lambda Expressions

## Goal
Master C++ Lambda Expressions to write concise, functional-style code for sorting, filtering, and transforming data collections.

## Learning Objectives
1.  Syntax: `[capture](params) -> return_type { body }`.
2.  Capture modes: `[=]` (value), `[&]` (reference), `[this]`, and specific captures.
3.  Use lambdas with STL algorithms (`std::sort`, `std::copy_if`, `std::transform`).

## Analogy: The Custom Robot Worker vs. The HR Hiring Process
*   **Old C++ (Functors/Function Pointers):** You need to sort a list of boxes.
    *   You have to leave the factory floor (the function).
    *   Go to HR (Global Scope).
    *   Write a job description (Struct/Class).
    *   Hire a worker (Instantiate).
    *   Bring them back to do the job.
    *   *Overkill for a 2-second task.*
*   **Modern C++ (Lambdas):** You build a tiny, disposable robot **right on the spot**.
    *   You tell it: "Hey, take these variables from my pocket (Capture), and sort these boxes based on weight."
    *   It does the job immediately and disappears.

## Practical Motivation
CV code involves lots of lists: detections, keypoints, matches.
*   "Sort matches by distance."
*   "Remove detections with low confidence."
*   "Transform points from Camera A to Camera B."

Writing a separate `struct Comparator` or function for every trivial operation is tedious. Lambdas let you define the logic **right where it happens**.

## Step-by-Step Instructions

### Task 1: Sorting with Lambdas
Open `src/main.cpp`. You have a `std::vector<Detection>`.
*   **Task:** Use `std::sort` with a lambda to sort detections by **confidence (descending)**.
    ```cpp
    std::sort(begin, end, [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    ```

### Task 2: Filtering with Captures
*   **Task:** Filter out detections below a certain threshold.
*   **Requirement:** The `threshold` variable is defined in `main`. You **must capture it** in the lambda.
    ```cpp
    float threshold = 0.5f;
    // [threshold] or [=]
    ```
    Use `std::copy_if` to move valid detections to a new vector.

## Common Pitfalls
*   **Dangling References:** `[&]` captures local variables by reference. If the lambda outlives the function (e.g., passed to a thread), it crashes. Use `[=]` for small primitives.
*   **Mutable Lambdas:** By default, `operator()` is const. Use `mutable` if you need to modify captured-by-value variables (rare).

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show sorted detections and then filtered detections.
