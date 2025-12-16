# Exercise 03: Color Spaces

## Goal
Implement manual RGB-to-Grayscale and RGB-to-HSV conversion kernels.

## Learning Objectives
1.  **BGR Order:** OpenCV loves Blue-Green-Red.
2.  **Luminance:** Why Green is worth more than Blue.
3.  **HSV:** Why color tracking is easier in HSV than RGB.

## Analogy: The Painter vs. The Printer
*   **RGB (The Printer):**
    *   "Mix 10% Red ink, 50% Green ink, 40% Blue ink."
    *   Hard to say "Make it brighter" without changing the color balance.
*   **HSV (The Painter):**
    *   **Hue:** "Pass me the Red paint." (The base color).
    *   **Saturation:** "Don't add any water." (Pure color vs washed out).
    *   **Value:** "Add some black." (Dark vs Bright).
    *   *Result:* If a shadow falls on a red ball, only the **Value** changes. The **Hue** stays the same. This makes tracking red balls easy.

## Practical Motivation
*   **Grayscale:** Reduces data by 2/3. Essential for geometry (edges, corners) where color is a distraction.
*   **HSV:** Essential for color segmentation (e.g., "Find the yellow tennis ball").

## Step-by-Step Instructions

### Task 1: BGR to Grayscale
Open `src/main.cpp`.
*   Implement `custom_gray(Mat& src, Mat& dst)`.
*   Formula: $Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$.
*   *Note:* `cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);`.
    *   `pixel[0]` is Blue.
    *   `pixel[1]` is Green.
    *   `pixel[2]` is Red.

### Task 2: BGR to HSV
*   Implement `custom_hsv(Mat& src, Mat& dst)`.
*   **Value:** $V = \max(R, G, B)$.
*   **Saturation:** $S = \frac{V - \min}{V}$.
*   **Hue:** Complicated formula (angle on color wheel).
    *   If $V = R$, $H = 60(G - B) / (V - \min)$.
    *   Etc.
*   *Important:* OpenCV maps Hue ($0^\circ..360^\circ$) to $0..180$ to fit in a byte (`uint8`).

### Task 3: Compare with OpenCV
*   Run `cv::cvtColor(src, dst, COLOR_BGR2GRAY)` and compare with your result.
*   Run `cv::cvtColor(src, dst, COLOR_BGR2HSV)` and compare.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
The program will display your converted images.
