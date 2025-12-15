# Exercise 08: Collision Checking (2D)

## Goal
Implement efficient intersection tests for geometric primitives.

## Learning Objectives
1.  **Primitives:** Circle vs Circle, AABB vs AABB, Circle vs AABB.
2.  **Separating Axis Theorem (SAT):** General concept for convex polygons.
3.  **Efficiency:** Fast rejection using bounding boxes.

## Practical Motivation
The core inner loop of any planner (RRT, PRM) is `IsCollisionFree()`. This function is called millions of times, so it must be fast.

## Step-by-Step Instructions
1.  Implement `struct Circle { float x, y, r; }`.
2.  Implement `struct Rect { float x, y, w, h; }`.
3.  Write functions:
    *   `check_circle_circle(c1, c2)`
    *   `check_rect_rect(r1, r2)`
    *   `check_circle_rect(c, r)`

## Verification
*   Unit tests covering intersecting, touching, and disjoint cases.
