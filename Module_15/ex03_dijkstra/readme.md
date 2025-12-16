# Exercise 03: Dijkstra vs A*

## Goal
Compare Dijkstra's Algorithm and A* Search side-by-side to visualize the difference in exploration.

## Learning Objectives
1.  **Dijkstra's Algorithm:** Understand that Dijkstra is a special case of A* where $h(n) = 0$.
2.  **Exploration Pattern:** Visualize how Dijkstra expands in all directions (uniform cost) vs A* which is guided.
3.  **Performance:** Observe that A* visits fewer nodes to find the same shortest path (if heuristic is admissible).

## Practical Motivation
If you don't know where the goal is (or if there are multiple goals), Dijkstra is perfect. If you know exactly where you want to go, A* is much faster.
**Analogy:**
*   **Dijkstra** is like dropping a stone in a pond. The ripples (search frontier) expand in a perfect circle until they hit the target.
*   **A*** is like a river flowing downhill towards the sea. It has a direction. It might meander around obstacles, but it generally pushes towards the destination.

## Step-by-Step Instructions
1.  **Reuse Grid:** Use the same grid map from Exercise 02.
2.  **Implement Dijkstra:** Copy your A* code, but set the heuristic function to always return 0.0.
3.  **Instrument Code:** Count the number of nodes "visited" (popped from the open set).
4.  **Compare:** Run both on the same map and print the "Nodes Visited" count.

## Todo
1.  Implement `Dijkstra::search` (or configure A* to be Dijkstra).
2.  Run the comparison benchmark.

## Verification
*   Dijkstra should visit MORE (or equal) nodes than A*.
*   Both should find the same path length (if optimal).
