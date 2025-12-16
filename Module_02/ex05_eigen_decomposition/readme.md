# Exercise 05: Eigen Decomposition (PCA)

## Goal
Compute principal axes of a 2D point cloud using Eigenvalues and Eigenvectors.

## Learning Objectives
1.  **Geometric Interpretation:** Eigenvectors = Directions of spread; Eigenvalues = Amount of spread.
2.  **Covariance Matrix:** How to build it from raw points.
3.  **Solver:** Using `Eigen::SelfAdjointEigenSolver` (Fast for symmetric matrices).

## Analogy: The Bee Swarm and the Box
*   **The Data:** A swarm of bees flying in a general direction.
*   **Eigenvectors:** The sides of the **Bounding Box** that perfectly fits the swarm.
    *   The longest side (Major Axis) points where the bees are going.
    *   The shortest side (Minor Axis) shows how tight the formation is.
*   **Eigenvalues:** The length of those sides.
    *   Big Eigenvalue = Very spread out.
    *   Small Eigenvalue = Very flat/thin.

## Practical Motivation
**Principal Component Analysis (PCA)** is fundamental in CV:
*   **Orientation Estimation:** Finding the rotation of an object in an image.
*   **Normal Estimation:** In 3D, the "flattest" direction (smallest eigenvalue) is the surface normal.
*   **Dimensionality Reduction:** Compressing face images (Eigenfaces).

## Step-by-Step Instructions

### Task 1: Generate Data
Open `src/main.cpp`.
*   We have a set of random 2D points stretched along a diagonal.
*   *Already implemented:* `generate_points()`.

### Task 2: Compute Mean and Center Data
1.  Compute the **Mean (Centroid)** of all points.
2.  Subtract the mean from every point to get **Centered Points**.
    *   *Why?* PCA requires data to be centered at (0,0).

### Task 3: Compute Covariance Matrix
1.  Compute the covariance matrix $\Sigma = \frac{1}{N-1} \sum (p_i)(p_i)^T$.
2.  Since points are centered, this is roughly $X X^T$ (if X is $2 \times N$).

### Task 4: Eigen Decomposition
1.  Use `Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance)`.
2.  Get eigenvalues: `solver.eigenvalues()`.
3.  Get eigenvectors: `solver.eigenvectors()`.
4.  **Crucial Note:** Eigen sorts results in **increasing order**.
    *   Index 0: Smallest Eigenvalue (Minor Axis).
    *   Index 1: Largest Eigenvalue (Major Axis).

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show the computed mean and the principal direction (which should roughly match the diagonal direction of the data).
