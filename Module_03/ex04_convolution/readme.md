# Exercise 04: Convolution

## Goal
Implement a 3x3 convolution from scratch and verify against `cv::filter2D`.

## Learning Objectives
1.  **Sliding Window:** How to iterate over an image and look at neighbors.
2.  **Padding:** Handling the borders (Zero vs Replicate).
3.  **Kernel Types:** Box filter, Gaussian, Sobel.

## Analogy: The Voting Booth (or The Survey)
*   **The Pixel:** A person standing in a grid of people.
*   **The Kernel:** A **Survey Form** with weights.
    *   *Box Filter (Blur):* "I want to know the average height of you and your 8 neighbors." (Everyone counts equally).
    *   *Sobel (Edge):* "I want to know the difference between your height and your neighbor's height." (If you are same height, result is 0. If different, result is big).
*   **Convolution:** You walk to every person, hold up the form, calculate the score, and write it down on a **new** sheet of paper (the output image).

## Practical Motivation
Convolution is the backbone of:
*   **Denoising:** Gaussian Blur.
*   **Edge Detection:** Sobel, Canny.
*   **Deep Learning:** CNNs are just stacks of convolutions learned from data.

## Step-by-Step Instructions

### Task 1: convolve
Open `src/main.cpp`.
*   Implement `custom_convolve(Mat& src, Mat& dst, Mat& kernel)`.
*   Assume `src` is `CV_8UC1` (Grayscale) and `kernel` is `3x3 float`.
*   Loop `y` from 1 to `rows-1`, `x` from 1 to `cols-1` (Skip borders for simplicity).
*   Inner Loop: `dy` from -1 to 1, `dx` from -1 to 1.
    *   `sum += src.at<uchar>(y+dy, x+dx) * kernel.at<float>(dy+1, dx+1)`.
*   **Important:** `sum` can be negative or > 255. Use `float` for sum.
*   Clamp/Saturate: `dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum)`.

### Task 2: Padding
*   The naive loop shrinks the image or leaves black borders.
*   Advanced (Optional): Use `cv::copyMakeBorder` to create a padded input, then loop over the original size.

### Task 3: Verify
*   Create a sharpening kernel:
    ```
     0 -1  0
    -1  5 -1
     0 -1  0
    ```
*   Run your function.
*   Run `cv::filter2D`.
*   Compare results.

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
The output images should look identical (except possibly at the 1-pixel border).
