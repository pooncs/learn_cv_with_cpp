# Exercise 08: Mesh Generation

## Goal
Create a mesh from points using Poisson Surface Reconstruction (conceptually) or a simpler Grid approach.

## Learning Objectives
1.  Understand implicit surface reconstruction.
2.  Implement Marching Cubes (conceptually) or simple Triangulation.
3.  For this exercise: Implement "Greedy Projection Triangulation" or simply mesh a grid of points.

## Theory & Background

### Greedy Projection
1.  Project points to a local 2D plane.
2.  Connect neighbors to form triangles.
3.  Ensure angles are not too small/large.

## Implementation Tasks

### Task 1: Simple Triangulation
Given an organized point cloud (grid structure $W \times H$), generate 2 triangles for every quad.

## Common Pitfalls
- Handling unorganized clouds is much harder (requires proper PSR).
- Organized cloud: Point at $(u, v)$ connects to $(u+1, v)$, $(u, v+1)$, etc.
