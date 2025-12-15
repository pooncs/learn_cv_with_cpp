# Exercise 08: Epipolar Geometry

## Goal
Compute the Fundamental Matrix ($F$) from point matches and visualize epipolar lines.

## Learning Objectives
1.  **Epipolar Constraint:** $x'^T F x = 0$. This implies that a point in one image corresponds to a line in the other.
2.  **Fundamental Matrix:** A $3 \times 3$ rank-2 matrix that encapsulates the intrinsic and extrinsic parameters of the stereo pair.
3.  **Visualization:** Drawing epipolar lines to verify the geometry.

## Practical Motivation
In stereo vision, searching for a match across the entire 2D image is slow. Knowing $F$ restricts the search to a 1D line (the epipolar line), drastically reducing computational cost and false positives.

## Theory
For a point $x$ in image 1, the corresponding point $x'$ in image 2 must lie on the line $l' = F x$.
Similarly, $l = F^T x'$.
The matrix $F$ can be estimated using the 8-point algorithm (or 7-point). RANSAC is again used to handle outliers.

## Step-by-Step Instructions

### Task 1: Find F
Use `cv::findFundamentalMat` with RANSAC.
-   Input: Matched points from two images.
-   Output: $F$ matrix and a status mask (inliers).

### Task 2: Compute Epipolar Lines
Use `cv::computeCorrespondEpilines`.
-   For points in Image 1, compute lines in Image 2.
-   For points in Image 2, compute lines in Image 1.

### Task 3: Visualization
Draw the lines on the images.
-   Line equation: $ax + by + c = 0$.
-   Draw a line from $x=0$ to $x=width$.

## Common Pitfalls
1.  **Normalization:** The 8-point algorithm is sensitive to coordinate scaling. OpenCV handles this internally, but be aware.
2.  **Planar Scenes:** If points lie on a plane, $F$ is degenerate. Use Homography instead.
3.  **Drawing:** Lines can be steep. Calculate intersection with image borders carefully.

## Verification
1.  Select a point in the left image.
2.  Observe the corresponding epipolar line in the right image passing through the matching feature.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
