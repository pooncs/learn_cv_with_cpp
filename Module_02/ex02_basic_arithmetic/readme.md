# Exercise 02: Basic Arithmetic

## Goal
Implement element-wise operations vs. matrix multiplication.

## Learning Objectives
1.  **Algebraic vs. Element-wise:** Know when to use `*` vs `.cwiseProduct()` (or `.array() * .array()`).
2.  **Broadcasting:** Efficiently add a vector to every column/row without loops.
3.  **Scaling:** Multiplying a matrix by a scalar.

## Analogy: The Mixer vs. The Cookie Cutter
*   **Matrix Multiplication (`A * B`):** Like a **Mixer**.
    *   Rows of A mix with Columns of B.
    *   The result is a transformation.
    *   Used for: Rotating points, changing coordinate systems.
*   **Element-wise Multiplication (`A.array() * B.array()`):** Like a **Cookie Cutter / Stencil**.
    *   You apply a value to the exact same spot in the other matrix.
    *   Used for: Masking images (Pixel A * Mask A), adjusting weights per-pixel.

## Practical Motivation
*   **Matrix Mult:** $y = Ax + b$ (Projecting a 3D point to 2D screen).
*   **Element-wise:** $I_{new} = I_{old} * Mask$ (Removing background from an image).

## Step-by-Step Instructions

### Task 1: Matrix Multiplication
Open `src/main.cpp`.
*   Initialize two $2 \times 2$ matrices $A$ and $B$.
*   Compute $C = A \times B$ using the `*` operator.
*   Print the result.

### Task 2: Element-wise Multiplication
*   Compute $D$ where $D_{ij} = A_{ij} \cdot B_{ij}$.
*   Use `.array()` view: `A.array() * B.array()`.
*   Alternatively, use `A.cwiseProduct(B)`.
*   Print the result. Notice how it differs from Task 1.

### Task 3: Broadcasting
*   Create a $3 \times 4$ matrix $M$.
*   Create a vector $v$ of size 3.
*   **Goal:** Add $v$ to **every column** of $M$.
*   **Method:** Use `.colwise() += v`.
*   Print the result.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
