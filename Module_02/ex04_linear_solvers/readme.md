# Exercise 04: Linear Solvers

## Goal
Solve systems of linear equations $Ax=b$ using various decompositions (LLT, LDLT, LU).

## Learning Objectives
1.  **Decomposition Choice:** Know when to use LLT (Fastest, strict requirements) vs LU/QR (General purpose).
2.  **Implementation:** Syntax `A.llt().solve(b)`.
3.  **Validation:** Checking if the solution is valid ($Ax \approx b$).

## Analogy: The Locksmiths
*   **The Problem ($Ax=b$):** You need to find the key code ($x$) that fits the Lock ($A$) to open the Door ($b$).
*   **LLT (Cholesky) - The Speed Runner:**
    *   Extremely fast.
    *   **Restriction:** Only works on high-end, perfectly symmetrical locks (SPD Matrices). If the lock is slightly bent (Not SPD), he fails completely.
*   **HouseholderQR - The Heavy Duty Pro:**
    *   Slower, carries heavy tools.
    *   **Advantage:** Can open almost any rusty, weird-shaped lock (General Matrices). Very stable.

## Practical Motivation
*   **Camera Calibration:** Solving for intrinsic parameters involves solving linear systems.
*   **Bundle Adjustment:** Solving normal equations $J^T J \Delta x = -J^T r$ (Often SPD, so LLT is king).
*   **Optical Flow:** Solving $A v = b$ for velocity.

## Step-by-Step Instructions

### Task 1: Setup System
Open `src/main.cpp`.
*   Create a $3 \times 3$ matrix $A$. Make it **Symmetric Positive Definite (SPD)**.
    *   *Tip:* Generate a random matrix $R$, then $A = R^T R$. This guarantees SPD.
*   Create a vector $b$.

### Task 2: LLT Solver (Cholesky)
*   Solve $Ax = b$ using `A.llt().solve(b)`.
*   Store result in $x_{llt}$.
*   Print $x_{llt}$.

### Task 3: LDLT Solver (Robust Cholesky)
*   Solve using `A.ldlt().solve(b)`.
*   Store in $x_{ldlt}$.
*   *Note:* LDLT is slightly slower than LLT but works for semi-definite matrices (where eigenvalues might be 0).

### Task 4: Verify
*   Compute error $e = A x - b$.
*   Print the norm of the error `e.norm()`. It should be close to 0.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
