# Exercise 10: Panorama Stitcher

## Goal
Combine all previous techniques (Keypoints, Descriptors, Matching, Homography, Warping) to build a simple image stitcher that creates a panorama from two images.

## Learning Objectives
1.  **Pipeline Integration:** Connect individual algorithms into a robust system.
2.  **Image Blending:** Learn how to blend overlapping regions to minimize seams (e.g., linear blending).
3.  **Coordinate Systems:** Manage the size of the final canvas to ensure both images fit.

## Practical Motivation
Panorama stitching is a classic CV application on every smartphone. While OpenCV has a high-level `Stitcher` class, building one from low-level components solidifies your understanding of the entire feature-based registration pipeline.

## Theory: The Pipeline
1.  **Detect & Describe:** Find features in Image 1 and Image 2 (ORB).
2.  **Match:** Find correspondence (BF Matcher + Ratio Test).
3.  **Register:** Compute Homography $H$ that maps Image 2 to Image 1.
4.  **Warp:** Transform Image 2 into Image 1's coordinate frame.
5.  **Composite:** Place Image 1 and the warped Image 2 onto a larger canvas.

## Step-by-Step Instructions

### Task 1: Feature Matching
Implement a function `computeStitchHomography(img1, img2)` that returns $H$.
-   Reuse logic from Ex 07.

### Task 2: Warping
Warp Image 2 using $H$.
-   **Trick:** The result might fall outside the original bounds. You might need to compute the bounding box of the warped corners and translate both images to fit in a new canvas. For simplicity, you can assume Image 1 is the center and Image 2 is warped "onto" it, possibly expanding to the right.

### Task 3: Blending
Create a mask where the images overlap.
-   Simple: Average the pixel values.
-   Better: Distance transform weighting (Multiband blending is advanced).
-   Simplest (for this exercise): Just overwrite or max-value.

## Common Pitfalls
1.  **Canvas Size:** If Image 2 is warped to the left of Image 1, you get negative coordinates. You need a translation matrix $T$ to shift everything into positive $(u,v)$.
2.  **Black Borders:** Warping introduces black pixels. Masking them out during blending is crucial.

## Verification
1.  Stitch `left.jpg` and `right.jpg`.
2.  Check alignment of edges.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
