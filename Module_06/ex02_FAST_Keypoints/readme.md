# Exercise 02: FAST Keypoint Detection

## Goal
Implement the FAST (Features from Accelerated Segment Test) corner detector manually. This detector is designed for high-performance real-time applications.

## Learning Objectives
1.  **Pixel Circle Test:** Understand the Bresenham circle of 16 pixels.
2.  **Intensity Thresholding:** Learn the logic of checking "brighter" and "darker" contiguous segments.
3.  **Optimization:** Understand why FAST is faster than Harris (early rejection).
4.  **Non-Maximum Suppression (NMS):** Apply NMS using a computed score function (e.g., sum of absolute differences).

## Practical Motivation
Harris is accurate but slow because of the convolution and eigenvalue calculation. For real-time SLAM (e.g., PTAM, ORB-SLAM), we need something extremely fast. FAST relies on simple intensity comparisons, making it very suitable for low-power devices.

## Theory: The FAST Test
For a pixel $p$ with intensity $I_p$, check the circle of 16 pixels around it (radius 3).

Pixel $p$ is a corner if there exists a set of $N$ contiguous pixels in the circle which are all:
-   Brighter than $I_p + t$, OR
-   Darker than $I_p - t$

Where $t$ is a threshold. Common values for $N$ are 12 (FAST-12), 9 (FAST-9).

### The Circle Indices (Radius 3)
Offsets relative to center (0,0):
0: (0, -3)
1: (1, -3)
2: (2, -2)
3: (3, -1)
4: (3, 0)
... and so on.

### High-Speed Test
To reject non-corners quickly, check pixels 1, 5, 9, 13 first. If at least 3 of these are not brighter/darker, $p$ cannot be a corner (for N=12).

## Step-by-Step Instructions

### Task 1: Pixel Access
Implement a helper to access the 16 pixels in the Bresenham circle efficiently.
-   **Hint:** Use `ptr<uchar>` for raw speed, or `at<uchar>` for safety.

### Task 2: The Segment Test
For each pixel in the image (excluding borders):
1.  Check if there are $N$ contiguous pixels satisfying the condition.
2.  Implement the early rejection optimization (check pixels 0, 4, 8, 12).

### Task 3: Score Function
To perform NMS, we need a "score" for the corner. A common score is the maximum threshold $t$ for which the pixel is still a corner, or the sum of absolute differences between the center and the pixels in the contiguous arc.
$$ V = \max( \sum (pixel - I_p - t), \sum (I_p - pixel - t) ) $$

### Task 4: Non-Maximum Suppression
If two adjacent pixels are detected as corners, keep the one with the higher score.

## Common Pitfalls
1.  **Boundary Checks:** Do not access pixels outside the image. Start loop from `y=3` to `rows-3`.
2.  **Contiguous Arc:** Handling the wrap-around (pixel 15 connects to pixel 0) is tricky.
3.  **Data Types:** Use `unsigned char` (uchar) for pixel values.

## Verification
1.  Build and run.
2.  Compare with `cv::FAST`.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
