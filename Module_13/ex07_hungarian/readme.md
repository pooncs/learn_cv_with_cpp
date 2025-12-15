# Exercise 07: Hungarian Algorithm (Data Association)

## Goal
Implement the Hungarian Algorithm (Munkres Algorithm) to solve the linear assignment problem. This is critical for associating tracked objects (from KF) with new detections.

## Learning Objectives
1.  Understand the Assignment Problem (Minimizing total cost).
2.  Implement the steps of the Hungarian Algorithm (or use a robust implementation).
3.  Handle rectangular cost matrices (unequal number of tracks vs detections).

## Practical Motivation
In a multi-object tracker, you have $M$ existing tracks and $N$ new detections. You calculate a Cost Matrix (e.g., Mahalanobis distance or IoU distance) of size $M \times N$. You need to find the one-to-one mapping that minimizes the total cost.

## Theory: Hungarian Algorithm Steps
1.  **Subtract Row Minima:** For each row, subtract the minimum value from all elements.
2.  **Subtract Col Minima:** For each col, subtract the minimum value.
3.  **Cover Zeros:** Find the minimum number of lines (rows/cols) to cover all zeros.
4.  **Optimality Test:** If lines == dim, done.
5.  **Create Zeros:** Find min uncovered value. Subtract from uncovered rows, add to covered cols. Repeat.

## Step-by-Step Instructions

### Task 1: Setup
1.  Define a class `Hungarian`.
2.  Input: `vector<vector<double>> costMatrix`.
3.  Output: `vector<int> assignment` (where `assignment[i] = j` means track `i` assigned to detection `j`, or -1 if unassigned).

### Task 2: Implementation
Implementing the full O(n^3) algorithm is complex. For this exercise, you can:
1.  Implement a **Greedy Nearest Neighbor** approach (simple baseline).
2.  **OR** Implement the full Hungarian Algorithm.
3.  **OR** Use a simplified version or a library-like structure.

*Recommendation:* Implement the full algorithm for completeness, as "Hungarian" is the title.

### Task 3: Test Cases
1.  Square Matrix (3x3).
2.  Rectangular Matrix (3 tracks, 5 detections).
3.  Rectangular Matrix (5 tracks, 3 detections).

## Common Pitfalls
-   **Rectangular Matrices:** The standard algorithm assumes square matrices. You must pad the matrix with dummy rows/cols having high cost (or 0 cost depending on formulation) to make it square.
-   **Maximization vs Minimization:** Hungarian minimizes cost. If using IoU (where higher is better), convert to cost ($1 - IoU$).

## Code Hints
A common C++ header-only implementation structure:
```cpp
class HungarianAlgorithm {
public:
    double Solve(vector<vector<double>>& DistMatrix, vector<int>& Assignment);
private:
    void assignmentoptimal(...);
    void buildassignmentvector(...);
    void computeassignmentcost(...);
    // ...
};
```

## Verification
Given a known cost matrix:
```
10, 19, 8, 15
10, 18, 7, 17
13, 16, 9, 14
12, 19, 8, 18
```
The optimal assignment should minimize sum.
