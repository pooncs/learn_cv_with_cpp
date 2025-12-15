# Exercise 02: Basic Arithmetic

## Goal
Implement element-wise operations vs. matrix multiplication.

## Learning Objectives
1.  Understand the distinction between algebraic matrix multiplication (`*`) and element-wise multiplication (`.array() * .array()`).
2.  Perform matrix addition, subtraction, and scaling.
3.  Implement broadcasting (adding a vector to each row/column).

## Practical Motivation
In equations like $y = Ax + b$, we use matrix multiplication.
However, in image processing, we often want to multiply two images pixel-by-pixel (e.g., masking), which is an element-wise operation.
Confusing these two is a very common bug.

## Theory & Background

### Matrix vs Array
Eigen separates linear algebra operations (Matrix) from coefficient-wise operations (Array).
- **Matrix Multiplication**: `m1 * m2` performs standard matrix product.
- **Element-wise**: To perform element-wise operations, you must convert the Matrix to an Array view using `.array()`.
  ```cpp
  Eigen::MatrixXd A, B;
  // ... init ...
  Eigen::MatrixXd C = A.array() * B.array(); // Element-wise product
  Eigen::MatrixXd D = A.cwiseProduct(B);     // Alternative syntax for element-wise
  ```

### Broadcasting
Broadcasting allows adding a vector to every column or row of a matrix.
```cpp
Eigen::MatrixXd M(2, 3);
Eigen::VectorXd v(2);
M.colwise() += v; // Add v to every column
```

## Implementation Tasks

### Task 1: Matrix Multiplication
Given two $2 \times 2$ matrices $A$ and $B$, compute $C = A \times B$.

### Task 2: Element-wise Multiplication
Given the same $A$ and $B$, compute $D$ where $D_{ij} = A_{ij} \cdot B_{ij}$.

### Task 3: Broadcasting
Create a $3 \times 4$ matrix and add a size-3 vector to each of its columns.

## Common Pitfalls
- Using `*` when you meant element-wise multiplication.
- Dimension mismatch in broadcasting (adding a size-4 vector to columns of size 3).
