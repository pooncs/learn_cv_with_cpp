# Exercise 03: Downsampling

## Goal
Implement Voxel Grid downsampling to reduce point cloud density.

## Learning Objectives
1.  Understand Voxel Grid filtering.
2.  Map float coordinates to integer voxel indices.
3.  Compute centroid of points within each voxel.

## Theory & Background

### Voxel Grid
Divide space into 3D cubes (voxels) of size $s$.
For a point $(x, y, z)$, the voxel index is:
$$ (i, j, k) = (\lfloor x/s \rfloor, \lfloor y/s \rfloor, \lfloor z/s \rfloor) $$

All points falling into the same $(i, j, k)$ are averaged to produce a single output point.

## Implementation Tasks

### Task 1: Voxel Map
Use `std::map<VoxelIndex, PointSum>` to accumulate points.
`VoxelIndex` can be a struct or a tuple.

### Task 2: Compute Centroids
Iterate through the map and compute average for each voxel.

## Common Pitfalls
- Efficient hashing of 3D integer coordinates.
- Integer overflow if indices are large.
