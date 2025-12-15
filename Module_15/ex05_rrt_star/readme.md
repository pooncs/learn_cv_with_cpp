# Exercise 05: RRT* (Optimized RRT)

## Goal
Implement RRT*, which asymptotically approaches the optimal path length.

## Learning Objectives
1.  **Rewiring:** The key difference from RRT. When adding a node, check if it can improve the path cost of its neighbors.
2.  **Asymptotic Optimality:** RRT* paths get straighter and shorter over time.

## Practical Motivation
RRT paths are jerky and inefficient. RRT* fixes this by constantly reorganizing the tree connectivity to minimize total path length.

## Step-by-Step Instructions
1.  Extend RRT logic.
2.  **Choose Parent:** When adding $q_{new}$, look at neighbors within radius $r$. Connect to the one that gives lowest total cost to start.
3.  **Rewire:** Check if $q_{new}$ can be a better parent for any of those neighbors. If so, rewire the edge.

## Verification
*   Run for 1000 iterations vs 5000 iterations. The path should visibly straighten out.
