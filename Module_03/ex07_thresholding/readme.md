# Exercise 07: Thresholding

## Goal
Implement global and adaptive (local mean) thresholding.

## Learning Objectives
1.  **Binarization:** Converting Grayscale to Black & White.
2.  **Global vs. Adaptive:** Why one size rarely fits all.
3.  **Integral Images (Concept):** How to calculate local averages fast.

## Analogy: The Standardized Test vs. Grading on a Curve
*   **Global Threshold ($T=128$):** "Anyone who scores above 50% passes."
    *   *Scenario:* A shadow falls on the paper (Lighting changes). Suddenly, everyone in the shadow area gets a "0" score because they are dark. They fail, even if there is clearly text there.
*   **Adaptive Threshold (The Curve):** "Anyone who scores higher than the average of their neighbors passes."
    *   *Scenario:* In the bright area, the average is 200. You need > 190 to pass (Text is dark).
    *   *Scenario:* In the shadow area, the average is 50. You need > 40 to pass.
    *   *Result:* You can read the text in both the sun and the shade.

## Practical Motivation
*   **Document Scanning:** Binarizing text from paper (CamScanner apps).
*   **Barcode Reading:** Finding the bars even if the lighting is uneven.

## Step-by-Step Instructions

### Task 1: Global Threshold
Open `src/main.cpp`.
*   Implement `custom_threshold(Mat& src, Mat& dst, double thresh)`.
*   Loop over pixels: If `val > thresh`, set `255`, else `0`.

### Task 2: Adaptive Threshold
*   Implement `custom_adaptive_threshold(Mat& src, Mat& dst, int blockSize, double C)`.
*   **Naive Approach:** For every pixel, loop over its neighbors to find the mean.
    *   `T = mean - C`.
    *   If `val > T`, set `255`.
*   **Efficient Approach (Hint):** Use `cv::blur` (Box Filter) to calculate the mean image first!
    *   `Mat mean_img; cv::blur(src, mean_img, Size(blockSize, blockSize));`.
    *   Then just compare `src(x,y)` with `mean_img(x,y) - C`.

### Task 3: Compare
*   Load an image with uneven lighting (`../../data/text_shadow.jpg` if available, or create a gradient image).
*   Compare Global vs Adaptive results.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
