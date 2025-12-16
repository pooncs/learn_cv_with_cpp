# Exercise 03: Downsampling

## Goal
Implement Voxel Grid downsampling to reduce point cloud density.

## Learning Objectives
1.  Understand Voxel Grid filtering.
2.  Map float coordinates to integer voxel indices.
3.  Compute centroid of points within each voxel.

## Analogy: The Minecraft Converter
*   **Real World:** Curves are smooth, and atoms are tiny.
*   **Minecraft (Voxel World):** Everything is made of big blocks.
*   **The Process:** To turn a real statue into a Minecraft statue:
    1.  Imagine a grid of 1x1x1 meter blocks over the statue.
    2.  If a block contains part of the statue, you place **one** block there.
    3.  **Averaging:** If the real statue had a nose, a chin, and a mouth all inside one block, they just become **one** block in the center.
    4.  **Result:** You still see the shape, but with way fewer "parts" (points).

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
