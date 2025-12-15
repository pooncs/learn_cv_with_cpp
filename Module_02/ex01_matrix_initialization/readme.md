# Exercise 01: Matrix Initialization

## Goal
Initialize dynamic and fixed-size Eigen matrices from arrays, standard containers, and files.

## Learning Objectives
By completing this exercise, you will be able to:
1.  Understand the difference between fixed-size (stack-allocated) and dynamic-size (heap-allocated) matrices in Eigen.
2.  Use the comma initializer syntax for quick matrix filling.
3.  Map standard C++ arrays or vectors to Eigen matrices without copying data.
4.  Read matrix data from a text file into an Eigen matrix.

## Practical Motivation
In Computer Vision and Robotics, matrices are ubiquitous. 
- **Fixed-size matrices** (like $3 \times 3$ rotation matrices or $4 \times 4$ transformation matrices) are small and performance-critical. Eigen optimizes these heavily using loop unrolling and stack allocation.
- **Dynamic-size matrices** (like an image represented as a matrix, or a large point cloud) are determined at runtime.
Knowing how to efficiently create and populate these structures is the first step to high-performance geometric computing.

## Theory & Background

### Eigen Basics
Eigen is a header-only C++ template library for linear algebra. The core class is `Eigen::Matrix<T, Rows, Cols>`.
- `Eigen::Matrix3f`: Float, 3x3 (Fixed)
- `Eigen::MatrixXd`: Double, Dynamic x Dynamic

### Initialization Methods
1.  **Comma Initializer**:
    ```cpp
    Eigen::Matrix3f m;
    m << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    ```
2.  **Map Interface**: Allows you to view a raw C-array as an Eigen matrix.
    ```cpp
    float data[] = {1, 2, 3, 4};
    Eigen::Map<Eigen::Matrix2f> m(data);
    ```
3.  **From File**: Typically involves reading a stream and parsing values.

## Implementation Tasks

### Task 1: Fixed-Size Initialization
Create a $4 \times 4$ float matrix representing an identity transformation.

### Task 2: Dynamic-Size Initialization
Allocate a matrix of size $rows \times cols$ (provided by user input or hardcoded) and fill it with random values.

### Task 3: Map from std::vector
Given a `std::vector<float>` containing 9 elements, map it to a $3 \times 3$ Eigen matrix **without copying**.

### Task 4: Read from File
Implement a function to read a matrix from `../../data/matrix.txt`. The file format is simple space-separated values.

## Common Pitfalls
- **Row-Major vs Column-Major**: Eigen defaults to **Column-Major** storage. Standard C/C++ 2D arrays are Row-Major. When mapping, be careful of the order.
- **Alignment**: Fixed-size vectorizable matrices (like `Vector4f`) require 16-byte alignment. If you use them as class members, you might need `EIGEN_MAKE_ALIGNED_OPERATOR_NEW`.

## Recommended Functions
- `Eigen::Matrix::Identity()`, `Eigen::Matrix::Zero()`, `Eigen::Matrix::Random()`
- `Eigen::Map<Type>(ptr)`
