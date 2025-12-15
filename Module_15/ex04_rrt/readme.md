# Exercise 04: Rapidly-exploring Random Trees (RRT)

## Goal
Implement RRT to find a path in a continuous 2D space (not a grid).

## Learning Objectives
1.  **Sampling-based Planning:** Unlike A* (grid-based), RRT works by randomly sampling the continuous space.
2.  **Tree Expansion:** `Nearest`, `Steer`, and `Extend` operations.
3.  **Probabilistic Completeness:** Understand that RRT finds a solution if one exists, given infinite time.

## Practical Motivation
For high-dimensional spaces (e.g., a 7-DOF robot arm), grids are impossible (curse of dimensionality). RRT is efficient in these high-dimensional spaces.

## Step-by-Step Instructions
1.  **Sample:** Pick a random point $q_{rand}$ in space.
2.  **Nearest:** Find the closest node $q_{near}$ in the existing tree.
3.  **Steer:** Move from $q_{near}$ towards $q_{rand}$ by a step size $\Delta$, creating $q_{new}$.
4.  **Collision Check:** If edge $(q_{near}, q_{new})$ is collision-free, add $q_{new}$ to tree.
5.  **Goal Check:** If $q_{new}$ is close to goal, terminate.

## Verification
*   The visualization should show a tree branching out from the start like lightning.
*   The path will look "jagged" and unoptimized (RRT is not optimal).
