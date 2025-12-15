# Exercise 05: Outlier Removal

## Goal
Implement Statistical Outlier Removal (SOR).

## Learning Objectives
1.  Compute mean distance to k-nearest neighbors for each point.
2.  Compute global mean $\mu$ and standard deviation $\sigma$ of these distances.
3.  Remove points with mean distance $> \mu + \alpha \cdot \sigma$.

## Theory & Background

### SOR Filter
Noise often appears as isolated points. By analyzing the distribution of neighbor distances, we can identify points that are "too far" from their neighbors.

## Implementation Tasks

### Task 1: Mean Distances
For every point, find k neighbors and compute average distance.

### Task 2: Filter
Calculate stats and filter.

## Common Pitfalls
- Efficient neighbor search (again).
- Handling small datasets.
