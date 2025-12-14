# Exercise 05: Morphology

## Goal
Implement Dilation and Erosion manually on binary images.

## Learning Objectives
1.  Understand Morphological operations as set operations.
2.  Implement min/max filters.
3.  Use structuring elements.

## Practical Motivation
- **Noise Removal**: Opening (Erosion then Dilation) removes small noise dots.
- **Hole Filling**: Closing (Dilation then Erosion) fills small holes.
- **Boundary Extraction**: Dilated - Eroded.

## Theory & Background

### Dilation
$$ (A \oplus B)(x, y) = \max_{(i, j) \in B} A(x+i, y+j) $$
Expands white regions.

### Erosion
$$ (A \ominus B)(x, y) = \min_{(i, j) \in B} A(x+i, y+j) $$
Shrinks white regions.

## Implementation Tasks

### Task 1: Dilate
Implement dilation using a $3 \times 3$ square structuring element (all 1s).
For each pixel, if any neighbor is 1 (255), the output is 1 (255).

### Task 2: Erode
Implement erosion.
For each pixel, if all neighbors are 1 (255), the output is 1 (255). Otherwise 0.

## Common Pitfalls
- Border handling (assume 0 for dilation, 1 for erosion to avoid growing/shrinking from border).
- In-place modification: You need a separate output buffer.
