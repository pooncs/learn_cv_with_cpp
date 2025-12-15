# Exercise 09: Octrees

## Goal
Implement a basic Octree for efficient spatial queries (e.g., radius search).

## Learning Objectives
1.  Understand Octree data structure (recursive subdivision).
2.  Insert points into the tree.
3.  Perform a radius search.

## Theory & Background

### Octree
A tree where each internal node has exactly 8 children.
Represents a cubic volume.
If a volume contains too many points, it splits into 8 sub-octants.

### Spatial Query
Instead of checking all $N$ points (linear scan), we only check points in relevant octants.

## Implementation Tasks

### Task 1: Octree Node
Define `struct OctreeNode`. Children pointers, bounding box, points (if leaf).

### Task 2: Insert
Recursively insert points.

### Task 3: Radius Search
Find all points within distance $R$ of query $Q$. Prune nodes that don't overlap with the query sphere.

## Common Pitfalls
- Memory management (lots of nodes).
- Bounding box intersection logic.
