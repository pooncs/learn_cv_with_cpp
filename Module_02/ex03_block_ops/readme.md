# Exercise 03: Block Operations

## Goal
Extract and manipulate Regions of Interest (ROI) using `.block()`, `.row()`, and `.col()`.

## Learning Objectives
1.  Understand how to access sub-matrices (blocks) in Eigen.
2.  Differentiate between fixed-size blocks (compile-time optimized) and dynamic blocks.
3.  Modify specific regions of a matrix in-place.

## Practical Motivation
In Computer Vision, we often need to:
- Crop an image (extract a sub-matrix).
- Copy a small patch into a larger image.
- Access the top-left $3 \times 3$ rotation part of a $4 \times 4$ pose matrix.

## Theory & Background

### The Block Method
- **Dynamic**: `.block(i, j, p, q)` starts at $(i, j)$ with size $p \times q$.
- **Fixed**: `.block<p, q>(i, j)` is faster if $p, q$ are known at compile time.

### Rows and Columns
- `.row(i)`: The i-th row.
- `.col(j)`: The j-th column.

### Vector Blocks
- `.head(n)`, `.tail(n)`, `.segment(i, n)` for vectors.

## Implementation Tasks

### Task 1: Extract 2x2 Block
Given a $4 \times 4$ matrix, extract the $2 \times 2$ block starting at $(1, 1)$.

### Task 2: Set Row to Zero
Set the 3rd row (index 2) of a matrix to zeros.

### Task 3: Paste Block
Create a small $2 \times 2$ matrix of ones and paste it into the bottom-right corner of a $4 \times 4$ matrix.

## Common Pitfalls
- **Indices**: 0-based indexing. `block(i, j, rows, cols)` args order.
- **Reference vs Copy**: `block()` returns an expression (view). Assigning it to a `MatrixXd` creates a copy. Assigning it to `Ref<MatrixXd>` or using it in an expression avoids copy.
