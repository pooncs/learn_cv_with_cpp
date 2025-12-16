# Exercise 01: Matrix Initialization

## Goal
Initialize dynamic and fixed-size Eigen matrices from arrays, standard containers, and files.

## Learning Objectives
1.  **Fixed vs. Dynamic:** Understand when to use `Matrix3f` (stack, fast) vs `MatrixXd` (heap, flexible).
2.  **Comma Initializer:** Quick syntax `m << 1, 2, 3...`.
3.  **Map Interface:** View raw memory (std::vector/array) as a Matrix without copying.
4.  **File I/O:** Parse matrix data from text files.

## Analogy: The Tic-Tac-Toe Grid vs. The Infinite Scroll
*   **Fixed-Size Matrix (`Matrix3f`):** Like a **Tic-Tac-Toe Board**.
    *   It is always 3x3.
    *   You can draw it on a napkin (Stack).
    *   The computer knows exactly where every cell is instantly. **Super Fast.**
*   **Dynamic-Size Matrix (`MatrixXd`):** Like an **Excel Spreadsheet**.
    *   You can keep adding rows forever.
    *   You have to ask the OS for a big chunk of office space (Heap) to store it.
    *   Slower, but necessary for things like "List of all points in a cloud".

## Practical Motivation
In CV, we use:
*   **Fixed Matrices:** For geometry. Rotation ($3 \times 3$), Transformation ($4 \times 4$), Camera Intrinsics ($3 \times 3$).
*   **Dynamic Matrices:** For data. Point Clouds ($N \times 3$), Images ($H \times W$), Feature Descriptors ($N \times 128$).

## Step-by-Step Instructions

### Task 1: Fixed-Size Initialization
Open `src/main.cpp`.
*   Create a `Eigen::Matrix4f` named `identity`.
*   Initialize it as an Identity matrix using `Eigen::Matrix4f::Identity()`.
*   Print it.

### Task 2: Dynamic-Size Initialization
*   Create a `Eigen::MatrixXd` named `randomMat`.
*   Resize it to $3 \times 2$.
*   Fill it with random values using `Eigen::MatrixXd::Random(rows, cols)`.

### Task 3: Map from std::vector (Zero-Copy)
You have a `std::vector<float> vec` with 9 elements.
*   **Goal:** Treat this vector as a $3 \times 3$ matrix.
*   **Challenge:** C++ vectors are Row-Major (usually), but Eigen is Column-Major by default.
*   **Task:** Use `Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>` to wrap the vector data.
*   *Note:* If you don't specify `RowMajor`, the matrix will look transposed because Eigen will read the data down columns instead of across rows.

### Task 4: Read from File
Open `src/matrix_utils.cpp`. Implement `readMatrix`.
1.  Open the file using `std::ifstream`.
2.  Read the entire file into a `std::vector<double>` first (since we don't know dimensions yet, or assuming a specific format).
    *   *Simplification:* For this task, assume the file contains a square matrix or just read it into a known size, OR simpler: read rows/cols from the first line if the format allows.
    *   *Robust Approach:* Read line by line. Count rows. Count numbers in first line for cols.
    *   Let's stick to the robust approach:
        *   Read lines into a `vector<vector<double>>`.
        *   Check that all rows have same width.
        *   Copy into `Eigen::MatrixXd`.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
