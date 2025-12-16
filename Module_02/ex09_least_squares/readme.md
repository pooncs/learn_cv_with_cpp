# Exercise 09: Least Squares Fitting

## Goal
Fit a plane to noisy 3D points using Singular Value Decomposition (SVD).

## Learning Objectives
1.  **Least Squares Formulation:** Turning a geometry problem into "Minimize Error".
2.  **SVD (Singular Value Decomposition):** The "Swiss Army Knife" of linear algebra.
3.  **Null Space:** Finding the solution to $Ax=0$ (Homogeneous systems).

## Analogy: The Best-Fit Table
*   **The Data:** A wobbly table with legs of slightly different lengths (Noisy points).
*   **The Plane:** The perfect flat sheet of glass you want to rest on top of them.
*   **Least Squares:** The math that calculates exactly how to tilt the glass so the average gap between the glass and the leg tips is minimized.
*   **SVD:** The tool that tells you: "This is the direction where the table is flattest."

## Practical Motivation
*   **Ground Plane Estimation:** Finding the floor in a robot's camera view.
*   **Wall Detection:** Fitting planes to LiDAR scans of a room.
*   **Calibration:** Finding the chessboard plane.

## Theory: SVD for Plane Fitting
A plane is $ax + by + cz + d = 0$.
The normal vector $n = [a, b, c]$ is the direction "least like" the spread of the data.
1.  **Center the data:** Subtract the mean. Now the plane passes through $(0,0,0)$, so $d=0$ relative to the center.
2.  **Form Matrix:** Stack coordinates into matrix $A$.
3.  **SVD:** The normal $n$ is the **Singular Vector** corresponding to the **Smallest Singular Value**. (It's the direction with the *least* variance).

## Step-by-Step Instructions

### Task 1: Generate Noisy Data
Open `src/main.cpp`.
*   We provide a function `generate_noisy_plane()` that creates points on $Z=0$ with some noise.

### Task 2: Compute Centroid
*   Calculate the mean point $\bar{p}$.
*   Subtract $\bar{p}$ from all points to get centered data $P_{centered}$.

### Task 3: Form Matrix & Compute SVD
*   Put centered points into a $3 \times N$ matrix (or $N \times 3$).
*   Compute SVD: `Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);`.
*   *Note:* If $A$ is $N \times 3$, the normal is the **last column of V**.
*   *Note:* If $A$ is $3 \times N$, the normal is the **last column of U** (or singular vector of $A A^T$). Let's assume $N \times 3$ layout for simplicity.

### Task 4: Recover Plane Equation
*   Normal $n = [a, b, c]$.
*   Recover $d$ using the original equation and the centroid: $a \bar{x} + b \bar{y} + c \bar{z} + d = 0 \implies d = -n \cdot \bar{p}$.
*   Print the equation. It should be close to $0x + 0y + 1z + 0 = 0$ (if generated on Z=0).

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
Output should show the fitted plane coefficients.
