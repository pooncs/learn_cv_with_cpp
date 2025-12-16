# Exercise 08: Gradients

## Goal
Compute Sobel X and Y derivatives and gradient magnitude/orientation.

## Learning Objectives
1.  **Derivatives:** How to find changes in intensity.
2.  **Magnitude:** How "strong" is the edge?
3.  **Orientation:** Which direction is the edge facing?

## Analogy: The Hiker's Map
*   **The Image:** A topographic map. Bright pixels are mountain peaks. Dark pixels are valleys.
*   **Gradient ($G_x, G_y$):** The slope of the ground where you are standing.
    *   $G_x$: Slope in the East-West direction.
    *   $G_y$: Slope in the North-South direction.
*   **Magnitude ($M = \sqrt{G_x^2 + G_y^2}$):** **Steepness**.
    *   High Magnitude = A Cliff (Edge).
    *   Low Magnitude = Flat ground.
*   **Orientation ($\theta = \text{atan2}(G_y, G_x)$):** **Compass Direction**.
    *   "Which way is uphill?"

## Practical Motivation
*   **Edge Detection:** Edges are just cliffs in the image intensity map.
*   **Feature Extraction:** SIFT and HOG descriptors are basically histograms of these "uphill directions".

## Step-by-Step Instructions

### Task 1: Sobel X and Y
Open `src/main.cpp`.
*   Compute $G_x$ and $G_y$.
*   Use `cv::Sobel(src, dst, ddepth, dx, dy, ksize)`.
*   **Critical:** Use `ddepth = CV_16S` or `CV_32F`. Why?
    *   If pixel goes from Bright (255) to Dark (0), the derivative is negative (-255).
    *   `uint8` cannot hold negative numbers. It will wrap around or clip.

### Task 2: Compute Magnitude
*   $M = \sqrt{G_x^2 + G_y^2}$.
*   Use `cv::magnitude(Gx, Gy, M)`. (Requires float inputs).
*   Convert $M$ back to `CV_8U` to visualize it (Normalize or clip).

### Task 3: Compute Orientation
*   $\theta = \text{atan2}(G_y, G_x)$.
*   Use `cv::phase(Gx, Gy, angle, true)` (true = degrees).

### Task 4: Visualization
*   Show $G_x$ (convert to 8U absolute value).
*   Show $G_y$.
*   Show Magnitude.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
