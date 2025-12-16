# Exercise 06: Big Data and Distributed Processing

## Goal
Understand the concepts of processing massive datasets that cannot fit into a single machine's memory, using concepts like MapReduce.

## Learning Objectives
1.  **Distributed Computing:** Why we need clusters for petabytes of data.
2.  **MapReduce:** The programming model of splitting tasks.
3.  **Spark/Hadoop:** The tools (conceptually).

## Practical Motivation
If you have 10,000 hours of video to analyze for autonomous driving, a single C++ program will take years. You need to split the video into 10,000 chunks and process them in parallel on 100 machines.

**Analogy:**
*   **Single Machine:** One person counting votes from the entire country.
*   **MapReduce:**
    *   **Map:** Each local polling station counts its own votes.
    *   **Reduce:** A central office sums up the totals from each station.

## Theory: MapReduce
*   **Map(k1, v1) -> list(k2, v2):** Takes input, produces intermediate key-value pairs. (e.g., Word -> 1)
*   **Reduce(k2, list(v2)) -> list(v2):** Aggregates values for the same key. (e.g., Word -> Sum)

## Step-by-Step Instructions

### Task 1: Conceptual MapReduce (C++ Simulation)
1.  Create a vector of strings (sentences).
2.  **Map Step:** Function that takes a string and returns a vector of `{word, 1}`.
3.  **Shuffle/Sort:** Group pairs by word.
4.  **Reduce Step:** Sum the counts for each word.

## Code Hints
```cpp
// Map
std::vector<std::pair<string, int>> map(string sentence) {
    // split and return {word, 1}
}

// Reduce
int reduce(string word, vector<int> counts) {
    // return sum(counts)
}
```

## Verification
Input: "Hello world", "Hello C++"
Output: Hello: 2, world: 1, C++: 1
