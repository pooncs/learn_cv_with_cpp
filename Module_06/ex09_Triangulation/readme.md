# Exercise 09: Triangulation

## Goal
Reconstruct 3D points from 2D stereo matches using Triangulation.

## Learning Objectives
1.  **Projection Matrices:** Understand $P = K [R | t]$.
2.  **Triangulation:** Solve for $X$ in equations $x = P X$ and $x' = P' X$.
3.  **Reprojection Error:** Measure the quality of the reconstruction.

## Practical Motivation
Once we know where the cameras are (Extrinsics $R, t$) and the camera properties (Intrinsics $K$), we can find the depth of any matched point. This is the core of Structure from Motion (SfM) and SLAM.

## Theory
Given two projection matrices $P_1$ and $P_2$, and a pair of matched points $(u_1, v_1)$ and $(u_2, v_2)$, we want to find $(X, Y, Z, W)$.
Since $x \times PX = 0$, we can form a system of linear equations $AX=0$ and solve using SVD (Direct Linear Transformation).
OpenCV provides `cv::triangulatePoints`.

## Step-by-Step Instructions

### Task 1: Setup Projection Matrices
Assume canonical camera 1: $P_1 = K [I | 0]$.
Assume camera 2 is translated by $t_x$: $P_2 = K [I | t_x]$.
Construct these $3 \times 4$ matrices.

### Task 2: Triangulate
1.  Convert points to $2 \times N$ matrices (floating point).
2.  Call `cv::triangulatePoints(P1, P2, pts1, pts2, points4D)`.

### Task 3: Convert to Euclidean
The output is Homogeneous coordinates $(x,y,z,w)$.
Convert to 3D: $(x/w, y/w, z/w)$.

## Common Pitfalls
1.  **Data Layout:** `triangulatePoints` expects $2 \times N$ input (channels=1) or vector of Point2f? No, it expects `InputArray` of size $2 \times N$ or $N \times 2$. Usually $2 \times N$ `CV_32F`.
2.  **W component:** If $w \approx 0$, the point is at infinity.
3.  **Coordinate Systems:** Ensure P matrices match the coordinate system of the points.

## Verification
1.  Create synthetic points at known depth $Z=10$.
2.  Project them to two cameras.
3.  Triangulate back and check error.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
