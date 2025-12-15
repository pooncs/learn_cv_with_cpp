# Exercise 03: Dijkstra vs A*

## Goal
Implement Dijkstra's algorithm and compare its performance (nodes visited) against A*.

## Learning Objectives
1.  **Dijkstra's Algorithm:** It's a special case of A* where $h(n) = 0$.
2.  **Performance Comparison:** Visualize which nodes are explored by both algorithms.
3.  **Admissibility:** Understand why Dijkstra is guaranteed to find the shortest path but might be slower.

## Practical Motivation
Understanding the trade-off between optimality and speed is crucial. Dijkstra explores in a circle (breadth-first-ish), while A* beelines for the goal.

## Step-by-Step Instructions
1.  Reuse your grid map and Node structure.
2.  Implement `Dijkstra` (same priority queue logic, but $f(n) = g(n)$).
3.  Run both on the same map.
4.  Count the number of nodes popped from the open set for both.

## Verification
*   Dijkstra should visit more (or equal) nodes than A*.
*   Both must find the same path length (if heuristic is admissible).
