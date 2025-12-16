# Exercise 06: Histograms

## Goal
Compute and visualize image histograms; implement histogram equalization.

## Learning Objectives
1.  **Histogram:** A frequency count of pixel values.
2.  **CDF (Cumulative Distribution Function):** The key to equalization.
3.  **Contrast Enhancement:** Making dark images visible.

## Analogy: The Wealth Redistribution
*   **The Image:** A society of pixels. Their brightness is their wealth ($0-$255).
*   **Low Contrast Image:** Everyone is middle class. Everyone has between $100 and $120. The image looks grey and boring.
*   **Histogram Equalization:** The government (Algorithm) enforces a new rule:
    *   "We will spread the wealth evenly."
    *   "The poorest 0.4% of people get value 0."
    *   "The next 0.4% get value 1."
    *   ...
    *   "The richest 0.4% get value 255."
*   **Result:** The society uses the full range of wealth ($0-$255). The image has **Maximum Contrast**.

## Practical Motivation
*   **Exposure Check:** Is the photo too dark (histogram bunched at 0) or blown out (bunched at 255)?
*   **Pre-processing:** Equalization makes features easier to detect for algorithms.

## Step-by-Step Instructions

### Task 1: Compute Histogram
Open `src/main.cpp`.
*   Implement `compute_histogram(Mat& src)`.
*   Create `int hist[256] = {0}`.
*   Loop over pixels: `hist[pixel_val]++`.

### Task 2: Visualize
*   Draw the histogram on a blank `512x256` image.
*   Normalize the histogram so the max value fits in the image height.
*   Draw lines from `(i, height)` to `(i, height - hist[i])`.

### Task 3: Equalize (Manual Implementation)
*   **CDF Calculation:**
    *   `cdf[0] = hist[0]`
    *   `cdf[i] = cdf[i-1] + hist[i]`
*   **Normalize:** `cdf[i] = cdf[i] * 255 / total_pixels`.
*   **Map:** Create output image where `dst(x,y) = cdf[src(x,y)]`.

### Task 4: Compare
*   Use `cv::equalizeHist`.
*   Compare your manual result with OpenCV's result.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
