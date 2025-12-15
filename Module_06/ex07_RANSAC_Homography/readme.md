# Exercise 07: RANSAC Homography

## Goal
Use RANSAC (Random Sample Consensus) to robustly estimate a Homography matrix between two images, even in the presence of outliers (wrong matches).

## Learning Objectives
1.  **Homography:** Understand the $3 \times 3$ matrix that maps points from one plane to another ($p' = H p$).
2.  **RANSAC Algorithm:** Understand the iterative process:
    *   Select random minimal subset (4 points).
    *   Compute model ($H$).
    *   Count inliers (points that fit $H$ with error $< \epsilon$).
    *   Keep best model.
3.  **Robust Matching:** See how RANSAC cleans up the noisy matches from the Ratio Test.

## Practical Motivation
Even with the Ratio Test, some matches are wrong. If we compute a Homography using Least Squares on all matches, the outliers will pull the solution away from the truth. RANSAC ignores outliers by finding the dominant consistent model.

## Theory: The Homography Matrix
$$
s \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$
It has 8 degrees of freedom (scale is arbitrary, so $h_{33}=1$). We need at least 4 point pairs to solve for it.

## Step-by-Step Instructions

### Task 1: Detect and Match
Use your previous tools (ORB + BF + Ratio Test) to get a set of "good" matches.

### Task 2: Prepare Data
Convert `std::vector<cv::DMatch>` and `std::vector<cv::KeyPoint>` into two `std::vector<cv::Point2f>` lists: `points1` and `points2`.

### Task 3: Find Homography
Use `cv::findHomography` with `cv::RANSAC`.
-   **Threshold:** The maximum reprojection error (e.g., 3.0 pixels) to consider a point an inlier.

### Task 4: Warp Perspective
Use the computed $H$ to warp Image 1 onto the perspective of Image 2 using `cv::warpPerspective`.

## Common Pitfalls
1.  **Not enough points:** If you have < 4 matches, RANSAC fails. Handle this gracefully.
2.  **Coordinate Type:** `cv::warpPerspective` expects the transformation matrix to be `CV_64F` (double).
3.  **Inverse Warping:** When warping, remember you are mapping *from* source *to* destination.

## Verification
1.  Take two photos of a flat object (book, poster) from different angles.
2.  Compute $H$ and warp the first image. It should look "rectified" or aligned with the second.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
