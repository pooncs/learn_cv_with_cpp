# Exercise 01: Harris Corner Detection

## Goal
Implement the Harris Corner Detection algorithm manually from scratch to understand the mathematical foundations of feature extraction.

## Learning Objectives
1.  **Gradients:** Understand how image gradients ($I_x, I_y$) capture local intensity changes.
2.  **Structure Tensor:** Learn how the second moment matrix (Structure Tensor) summarizes local gradient distribution.
3.  **Corner Response:** Implement the Harris response function $R = \det(M) - k \cdot \text{trace}(M)^2$.
4.  **Non-Maximum Suppression (NMS):** Refine detections by keeping only local maxima.

## Practical Motivation
Corner detection is the first step in many Computer Vision pipelines, including:
-   **Image Stitching:** Finding matching points between overlapping images.
-   **Tracking:** Identifying stable points to track over video frames (KLT tracker).
-   **3D Reconstruction:** Finding corresponding points in stereo images.

While OpenCV provides `cv::cornerHarris`, implementing it manually gives you deep insight into *why* some corners are better than others and how noise affects detection.

## Theory: The Mathematics of Corners
A corner is a region where the image intensity changes in *all* directions. We can detect this by looking at the **Structure Tensor** (or Second Moment Matrix) $M$ for a window $W$ around a pixel $(u, v)$:

$$
M = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
$$

Where:
-   $I_x, I_y$ are image derivatives in x and y directions.
-   $w(x,y)$ is a window function (usually a Gaussian).

The eigenvalues $\lambda_1, \lambda_2$ of $M$ characterize the region:
-   **Flat:** $\lambda_1 \approx 0, \lambda_2 \approx 0$
-   **Edge:** $\lambda_1 \gg 0, \lambda_2 \approx 0$ (or vice versa)
-   **Corner:** $\lambda_1 \gg 0, \lambda_2 \gg 0$

Instead of computing eigenvalues (which is expensive), Harris proposed a response score $R$:

$$
R = \det(M) - k \cdot \text{trace}(M)^2
$$

$$
\det(M) = \lambda_1 \lambda_2 = I_x^2 I_y^2 - (I_x I_y)^2
$$
$$
\text{trace}(M) = \lambda_1 + \lambda_2 = I_x^2 + I_y^2
$$

$k$ is a sensitivity constant, typically $0.04 - 0.06$.

## Step-by-Step Instructions

### Task 1: Compute Gradients
Compute the horizontal ($I_x$) and vertical ($I_y$) derivatives of the input grayscale image.
-   **Hint:** Use `cv::Sobel`.

### Task 2: Compute Products
Calculate the three unique components of the structure tensor for every pixel:
-   $I_{xx} = I_x \cdot I_x$
-   $I_{yy} = I_y \cdot I_y$
-   $I_{xy} = I_x \cdot I_y$

### Task 3: Compute Structure Tensor (Window Sum)
Apply a Gaussian blur to $I_{xx}, I_{yy}, I_{xy}$. This effectively performs the weighted sum over the window $W$.
-   **Hint:** Use `cv::GaussianBlur`.

### Task 4: Compute Harris Response
For each pixel, calculate $R$ using the formula above. Store the result in a floating-point image (e.g., `CV_32F`).

### Task 5: Thresholding and NMS
1.  Threshold the response map: Keep pixels where $R > \text{threshold}$.
2.  (Optional but recommended) Perform Non-Maximum Suppression: A pixel is a corner only if it is the local maximum in a $3 \times 3$ neighborhood.

## Common Pitfalls
1.  **Data Types:** Gradients can be negative. Use `CV_16S` or `CV_32F` for derivatives. Do not use `CV_8U`.
2.  **Sensitivity:** If $k$ is too large, you detect fewer corners. If too small, you detect edges as corners.
3.  **Normalization:** The response values can be very large. It helps to normalize the result to $0..255$ for visualization, but perform thresholding on the raw values (or consistently normalized ones).

## Verification
1.  Build the project.
2.  Run the tests.
3.  Compare your output visually with `cv::cornerHarris`.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
